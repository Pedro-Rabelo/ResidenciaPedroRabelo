import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.mobilenetv3_multitask import mobilenetv3_large_multitask
from utils.metrics import MarginCosineProduct
from utils.dataset_landmarks import ImageFolderWithLandmarks, HybridDataset, create_validation_split_with_landmarks
from utils.multitask_loss import MultiTaskLossAdvanced
from utils.general import (
    setup_seed,
    AverageMeter,
    LOGGER,
    save_on_master
)
import evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description="VGGFace2 Multi-task Training with Anti-Spoofing")
    
    # Dataset and Paths
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Path to VGGFace2 aligned images'
    )
    parser.add_argument(
        '--landmarks-json',
        type=str,
        required=True,
        help='Path to landmarks JSON file'
    )
    
    parser.add_argument(
        '--casia-root',
        type=str,
        default=None,
        help='Path to CASIA-FASD dataset (optional, enables anti-spoofing)'
    )
    parser.add_argument(
        '--spoofing-json',
        type=str,
        default=None,
        help='Path to spoofing labels JSON for VGGFace2 (optional)'
    )
    parser.add_argument(
        '--use-hybrid-dataset',
        action='store_true',
        help='Use hybrid dataset (VGGFace2 + CASIA-FASD)'
    )
    parser.add_argument(
        '--casia-ratio',
        type=float,
        default=0.3,
        help='Ratio of CASIA samples in hybrid dataset (default: 0.3)'
    )
    
    # Model Settings
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=512,
        help='Embedding dimension (default: 512)'
    )
    
    # Training Hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training (default: 256)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Initial learning rate (default: 0.1)'
    )
    
    # Learning Rate Scheduler
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='+',
        default=[10, 20, 25],
        help='Epochs to reduce learning rate (default: [10, 20, 25])'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Learning rate decay factor (default: 0.1)'
    )
    
    # Optimizer
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay (default: 5e-4)'
    )
    
    # Multi-task Learning
    parser.add_argument(
        '--landmark-weight',
        type=float,
        default=0.5,
        help='Weight for landmark loss (default: 0.5)'
    )
    parser.add_argument(
        '--spoofing-weight',
        type=float,
        default=0.3,
        help='Weight for anti-spoofing loss (default: 0.3)'
    )
    parser.add_argument(
        '--use-wing-loss',
        action='store_true',
        help='Use Wing Loss for landmarks instead of SmoothL1'
    )
    
    # Dataset Split
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training split ratio (default: 0.8 = 80%% train, 20%% val)'
    )
    parser.add_argument(
        '--min-images-per-class',
        type=int,
        default=2,
        help='Minimum images per identity to keep (default: 2)'
    )
    
    # Paths
    parser.add_argument(
        '--save-path',
        type=str,
        default='weights/vggface2',
        help='Path to save model checkpoints'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    # Training
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of data loader workers (default: 8)'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Print frequency in batches (default: 100)'
    )
    
    # LFW Evaluation
    parser.add_argument(
        '--lfw-root',
        type=str,
        default='data/val',
        help='Path to LFW dataset for validation (default: data/val)'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=1,
        help='LFW evaluation frequency in epochs (default: 1)'
    )
    
    return parser.parse_args()


def train_one_epoch_multitask(
    model,
    classification_head,
    criterion_multitask,
    optimizer,
    data_loader,
    device,
    epoch,
    params
):
    """Training loop for one epoch with anti-spoofing"""
    model.train()
    classification_head.train()
    
    losses_total = AverageMeter("Total Loss", ":6.3f")
    losses_cls = AverageMeter("Cls Loss", ":6.3f")
    losses_landmark = AverageMeter("Landmark Loss", ":6.3f")
    losses_spoofing = AverageMeter("Spoof Loss", ":6.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    spoofing_acc_meter = AverageMeter("Spoof Acc", ":4.2f")
    batch_time = AverageMeter("Time", ":4.3f")
    
    start_time = time.time()
    last_batch_idx = len(data_loader) - 1
    
    for batch_idx, batch_data in enumerate(data_loader):
        last_batch = last_batch_idx == batch_idx
        
        images, targets, landmarks_gt, is_spoof = batch_data
        
        # Move to device
        images = images.to(device)
        targets = targets.to(device)
        landmarks_gt = landmarks_gt.to(device)
        is_spoof = is_spoof.to(device)  # NOVO
        
        # Zero gradients
        optimizer.zero_grad()
        
        embeddings, landmarks_pred, spoofing_pred = model(
            images, 
            return_landmarks=True,
            return_spoofing=True
        )
        
        # Classification via MCP (apenas para samples com identidade válida)
        # CASIA samples têm identity=-1, então precisamos filtrar
        valid_identity_mask = targets >= 0
        
        if valid_identity_mask.sum() > 0:
            # Apenas calcula classification loss para VGGFace2 samples
            cls_output = classification_head(
                embeddings[valid_identity_mask], 
                targets[valid_identity_mask]
            )
            
            total_loss, loss_dict = criterion_multitask(
                cls_output,
                landmarks_pred[valid_identity_mask],
                spoofing_pred,  # Todos os samples têm label de spoofing
                targets[valid_identity_mask],
                landmarks_gt[valid_identity_mask],
                is_spoof
            )
            
            # Calculate classification accuracy (apenas VGGFace2)
            _, predicted = torch.max(cls_output.data, 1)
            accuracy = (predicted == targets[valid_identity_mask]).float().mean() * 100
        else:
            # Batch contém apenas CASIA samples (sem identidade)
            # Apenas calcula spoofing loss
            spoofing_gt_float = is_spoof.float().unsqueeze(1)
            spoofing_loss_fn = torch.nn.BCEWithLogitsLoss()
            total_loss = spoofing_loss_fn(spoofing_pred, spoofing_gt_float)
            
            loss_dict = {
                'total': total_loss.item(),
                'classification': 0.0,
                'landmark': 0.0,
                'spoofing': total_loss.item()
            }
            accuracy = torch.tensor(0.0)
        
        # Calculate spoofing accuracy (para todos os samples)
        spoofing_probs = torch.sigmoid(spoofing_pred).squeeze()
        spoofing_preds = (spoofing_probs > 0.5).long()
        spoofing_accuracy = (spoofing_preds == is_spoof).float().mean() * 100
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        losses_total.update(loss_dict['total'], images.size(0))
        losses_cls.update(loss_dict['classification'], images.size(0))
        losses_landmark.update(loss_dict['landmark'], images.size(0))
        losses_spoofing.update(loss_dict['spoofing'], images.size(0)) 
        accuracy_meter.update(accuracy.item(), images.size(0))
        spoofing_acc_meter.update(spoofing_accuracy.item(), images.size(0)) 
        batch_time.update(time.time() - start_time)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Log progress
        if batch_idx % params.print_freq == 0 or last_batch:
            lr = optimizer.param_groups[0]['lr']
            log = (
                f'Epoch: [{epoch}/{params.epochs}][{batch_idx:05d}/{len(data_loader):05d}] '
                f'Loss: {losses_total.avg:6.3f} '
                f'(Cls: {losses_cls.avg:6.3f}, Lmk: {losses_landmark.avg:6.3f}, '
                f'Spf: {losses_spoofing.avg:6.3f}) ' 
                f'Acc: {accuracy_meter.avg:4.2f}% '
                f'SpfAcc: {spoofing_acc_meter.avg:4.2f}% '
                f'LR: {lr:.5f} '
                f'Time: {batch_time.avg:4.3f}s'
            )
            LOGGER.info(log)
    
    # End-of-epoch summary
    log = (
        f'Epoch [{epoch}/{params.epochs}] Summary: '
        f'Total Loss: {losses_total.avg:6.3f}, '
        f'Cls: {losses_cls.avg:6.3f}, '
        f'Lmk: {losses_landmark.avg:6.3f}, '
        f'Spf: {losses_spoofing.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%, '
        f'Spoof Acc: {spoofing_acc_meter.avg:4.2f}%'
    )
    LOGGER.info(log)


def validate_lfw(model, device, lfw_root='data/val'):
    """Validates model on LFW dataset"""
    try:
        model.eval()
        accuracy, _ = evaluate.eval(model, device=device, lfw_root=lfw_root)
        model.train()
        return accuracy
    except Exception as e:
        LOGGER.warning(f"LFW validation failed: {e}")
        return 0.0


def main(params):
    # Setup
    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LOGGER.info("="*70)
    LOGGER.info("VGGFACE2 TRAINING - MULTI-TASK WITH ANTI-SPOOFING")
    LOGGER.info("="*70)
    LOGGER.info(f"Device: {device}")
    LOGGER.info(f"Root: {params.root}")
    LOGGER.info(f"Landmarks: {params.landmarks_json}")
    
    if params.casia_root:
        LOGGER.info(f"CASIA-FASD: {params.casia_root}")
        LOGGER.info(f"Anti-Spoofing: ENABLED")
        LOGGER.info(f"Spoofing weight: {params.spoofing_weight}")
        if params.use_hybrid_dataset:
            LOGGER.info(f"Using HYBRID dataset (CASIA ratio: {params.casia_ratio})")
    else:
        LOGGER.info(f"Anti-Spoofing: DISABLED (no CASIA-FASD)")
    
    LOGGER.info(f"Batch size: {params.batch_size}")
    LOGGER.info(f"Epochs: {params.epochs}")
    LOGGER.info("="*70 + "\n")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    LOGGER.info("Loading dataset...")
    
    if params.use_hybrid_dataset and params.casia_root:
        # Dataset híbrido (VGGFace2 + CASIA-FASD)
        full_dataset = HybridDataset(
            vggface_root=params.root,
            landmarks_json=params.landmarks_json,
            casia_root=params.casia_root,
            transform=train_transform,
            casia_ratio=params.casia_ratio
        )
    else:
        # Dataset padrão (só VGGFace2, com ou sem spoofing labels)
        full_dataset = ImageFolderWithLandmarks(
            root=params.root,
            landmarks_json=params.landmarks_json,
            spoofing_json=params.spoofing_json,
            transform=train_transform
        )
    
    # Get actual number of classes
    num_classes = full_dataset.get_num_classes()
    LOGGER.info(f"Dataset loaded: {num_classes:,} classes\n")
    
    # Split into train and validation
    val_split = 1.0 - params.train_split
    train_dataset, val_dataset = create_validation_split_with_landmarks(
        full_dataset, 
        val_split=val_split
    )
    
    LOGGER.info(f"Training samples: {len(train_dataset):,}")
    LOGGER.info(f"Validation samples: {len(val_dataset):,}")
    LOGGER.info(f"Split ratio: {params.train_split*100:.1f}% / {val_split*100:.1f}%\n")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Model with multi-task learning
    LOGGER.info("Creating MobileNetV3-Large with multi-task learning + anti-spoofing...")
    model = mobilenetv3_large_multitask(embedding_dim=params.embedding_dim).to(device)
    
    # Classification head (MCP - CosFace)
    classification_head = MarginCosineProduct(
        in_features=params.embedding_dim,
        out_features=num_classes,
        s=30.0,
        m=0.40
    ).to(device)
    
    LOGGER.info(f"Model created: {params.embedding_dim}D embeddings, {num_classes:,} classes")
    
    # Multi-task loss with anti-spoofing
    criterion_multitask = MultiTaskLossAdvanced(
        landmark_weight=params.landmark_weight,
        spoofing_weight=params.spoofing_weight,
        use_wing_loss=params.use_wing_loss
    )
    
    loss_type = "Wing Loss" if params.use_wing_loss else "SmoothL1 Loss"
    LOGGER.info(f"Loss: Classification + Landmark ({loss_type}) + Anti-Spoofing")
    LOGGER.info(f"Weights: landmark={params.landmark_weight}, spoofing={params.spoofing_weight}\n")
    
    # Optimizer
    optimizer = torch.optim.SGD(
        [
            {'params': model.parameters()},
            {'params': classification_head.parameters()}
        ],
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=params.milestones,
        gamma=params.gamma
    )
    
    LOGGER.info(f"Optimizer: SGD (lr={params.lr}, momentum={params.momentum}, wd={params.weight_decay})")
    LOGGER.info(f"LR Schedule: MultiStepLR (milestones={params.milestones}, gamma={params.gamma})\n")
    
    # Resume from checkpoint
    start_epoch = 0
    best_lfw_accuracy = 0.0
    
    if params.checkpoint and os.path.isfile(params.checkpoint):
        LOGGER.info(f"Resuming from checkpoint: {params.checkpoint}")
        ckpt = torch.load(params.checkpoint, map_location="cpu")
        
        model.load_state_dict(ckpt['model'])
        classification_head.load_state_dict(ckpt['classification_head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        start_epoch = ckpt['epoch']
        best_lfw_accuracy = ckpt.get('best_lfw_accuracy', 0.0)
        
        LOGGER.info(f"Resumed from epoch {start_epoch}, best LFW: {best_lfw_accuracy:.4f}\n")
    
    # Create save directory
    os.makedirs(params.save_path, exist_ok=True)
    
    # Training loop
    LOGGER.info("="*70)
    LOGGER.info("STARTING TRAINING")
    LOGGER.info("="*70 + "\n")
    
    for epoch in range(start_epoch, params.epochs):
        # Train one epoch
        train_one_epoch_multitask(
            model,
            classification_head,
            criterion_multitask,
            optimizer,
            train_loader,
            device,
            epoch,
            params
        )
        
        # Step scheduler
        lr_scheduler.step()
        
        # Save last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'classification_head': classification_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_lfw_accuracy': best_lfw_accuracy,
            'num_classes': num_classes,
            'args': params
        }
        
        last_save_path = os.path.join(
            params.save_path,
            'mobilenetv3_vggface2_multitask_antispoofing_last.ckpt'
        )
        save_on_master(checkpoint, last_save_path)
        
        # LFW Evaluation
        if (epoch + 1) % params.eval_freq == 0:
            LOGGER.info(f"\nEvaluating on LFW (epoch {epoch+1})...")
            lfw_accuracy = validate_lfw(model, device, params.lfw_root)
            LOGGER.info(f"LFW Accuracy: {lfw_accuracy:.4f}\n")
            
            # Save best model
            if lfw_accuracy > best_lfw_accuracy:
                best_lfw_accuracy = lfw_accuracy
                checkpoint['best_lfw_accuracy'] = best_lfw_accuracy
                
                best_save_path = os.path.join(
                    params.save_path,
                    'mobilenetv3_vggface2_multitask_antispoofing_best.ckpt'
                )
                save_on_master(checkpoint, best_save_path)
                
                LOGGER.info(f"✅ New best LFW accuracy: {best_lfw_accuracy:.4f}")
                LOGGER.info(f"✅ Best model saved to: {best_save_path}\n")
    
    # Training completed
    LOGGER.info("="*70)
    LOGGER.info("TRAINING COMPLETED")
    LOGGER.info("="*70)
    LOGGER.info(f"Best LFW accuracy: {best_lfw_accuracy:.4f}")
    LOGGER.info(f"Models saved in: {params.save_path}")
    LOGGER.info("="*70)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)