import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import json

from models.mobilenetv3_multitask import mobilenetv3_large_multitask
from utils.spoofing_dataset import CASIAFASDDataset


def convert_to_python_types(obj):
    """Converte numpy types para Python types para serialização JSON"""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class LFWDataset(Dataset):
    """
    Dataset loader para LFW pairs
    
    Estrutura esperada:
        lfw_root/
        ├── person1/
        │   ├── person1_0001.jpg
        │   └── person1_0002.jpg
        └── person2/
            └── person2_0001.jpg
        
        pairs.txt (no diretório pai de lfw_root)
    """
    
    def __init__(self, root, pairs_file=None, transform=None):
        """
        Args:
            root: Path para a pasta LFW (contém as pastas de pessoas)
            pairs_file: Path para pairs.txt (se None, procura no parent dir)
            transform: Transformações da imagem
        """
        self.root = Path(root)
        
        # ========== Trata o caminho do LFW corretamente ==========
        if self.root.name == 'lfw' and self.root.exists():
            self.lfw_dir = self.root
        elif (self.root / 'lfw').exists():
            self.lfw_dir = self.root / 'lfw'
        else:
            self.lfw_dir = self.root
        
        # Procura pairs.txt
        if pairs_file is None:
            if (self.lfw_dir / 'pairs.txt').exists():
                self.pairs_file = self.lfw_dir / 'pairs.txt'
            elif (self.lfw_dir.parent / 'pairs.txt').exists():
                self.pairs_file = self.lfw_dir.parent / 'pairs.txt'
            else:
                raise FileNotFoundError(f"pairs.txt not found in {self.lfw_dir} or {self.lfw_dir.parent}")
        else:
            self.pairs_file = Path(pairs_file)
        
        self.transform = transform
        
        # Valida estrutura
        if not self.lfw_dir.exists():
            raise FileNotFoundError(f"LFW directory not found: {self.lfw_dir}")
        
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")
        
        print(f"LFW directory: {self.lfw_dir}")
        print(f"Pairs file: {self.pairs_file}")
        
        # Carrega pares
        self.pairs = []
        self.labels = []
        self._load_pairs()
        
        print(f"LFW Dataset loaded: {len(self.pairs)} pairs")
    
    def _load_pairs(self):
        """Carrega pares do arquivo pairs.txt"""
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()
        
        # Primeira linha contém número de folds e pares por fold
        first_line = lines[0].strip().split()
        if len(first_line) == 2:
            n_folds, n_pairs = map(int, first_line)
        else:
            # Formato alternativo: apenas número de pares
            n_folds = 10
            n_pairs = len(lines) // 20  # Estimativa
        
        idx = 1
        for fold in range(n_folds):
            # Pares positivos (mesma pessoa)
            for _ in range(n_pairs):
                if idx >= len(lines):
                    break
                
                parts = lines[idx].strip().split()
                if len(parts) < 3:
                    idx += 1
                    continue
                
                name = parts[0]
                img1_idx = int(parts[1])
                img2_idx = int(parts[2])
                
                img1_path = self.lfw_dir / name / f"{name}_{img1_idx:04d}.jpg"
                img2_path = self.lfw_dir / name / f"{name}_{img2_idx:04d}.jpg"
                
                # Só adiciona se ambas existem
                if img1_path.exists() and img2_path.exists():
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(1)  # Mesma pessoa
                
                idx += 1
            
            # Pares negativos (pessoas diferentes)
            for _ in range(n_pairs):
                if idx >= len(lines):
                    break
                
                parts = lines[idx].strip().split()
                if len(parts) < 4:
                    idx += 1
                    continue
                
                name1 = parts[0]
                img1_idx = int(parts[1])
                name2 = parts[2]
                img2_idx = int(parts[3])
                
                img1_path = self.lfw_dir / name1 / f"{name1}_{img1_idx:04d}.jpg"
                img2_path = self.lfw_dir / name2 / f"{name2}_{img2_idx:04d}.jpg"
                
                # Só adiciona se ambas existem
                if img1_path.exists() and img2_path.exists():
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(0)  # Pessoas diferentes
                
                idx += 1
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Carrega imagens
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load images: {img1_path}, {img2_path}: {e}")
        
        # Aplica transformações
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def extract_features(model, dataloader, device):
    """
    Extrai features de todos os pares
    
    Returns:
        features1: features da primeira imagem de cada par
        features2: features da segunda imagem de cada par
        labels: labels dos pares
    """
    model.eval()
    
    features1_list = []
    features2_list = []
    labels_list = []
    
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Extracting features"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Extrai features (sem landmarks e sem spoofing score)
            feat1 = model.extract_features(img1, return_spoofing_score=False)
            feat2 = model.extract_features(img2, return_spoofing_score=False)
            
            features1_list.append(feat1.cpu().numpy())
            features2_list.append(feat2.cpu().numpy())
            labels_list.append(label.numpy())
    
    features1 = np.vstack(features1_list)
    features2 = np.vstack(features2_list)
    labels = np.concatenate(labels_list)
    
    return features1, features2, labels


def compute_similarity(features1, features2):
    """
    Calcula similaridade coseno entre pares de features
    
    Returns:
        similarities: array de similaridades [-1, 1]
    """
    # Normaliza features
    features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    
    # Similaridade coseno
    similarities = np.sum(features1_norm * features2_norm, axis=1)
    
    return similarities


def compute_metrics(similarities, labels, output_dir='results'):
    """
    Calcula métricas de avaliação e gera gráficos
    
    Returns:
        accuracy: acurácia no melhor threshold
        best_threshold: melhor threshold encontrado
        auc_score: área sob a curva ROC
    """
    # Calcula ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)
    
    # Encontra melhor threshold (maximiza accuracy)
    accuracies = []
    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        accuracies.append(accuracy)
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    # Calcula TAR @ FAR
    tar_at_far = {}
    for far_target in [0.001, 0.01, 0.1]:
        idx = np.where(fpr <= far_target)[0]
        if len(idx) > 0:
            tar_at_far[far_target] = tpr[idx[-1]]
        else:
            tar_at_far[far_target] = 0.0
    
    # Cria output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plota ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - LFW Evaluation', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = Path(output_dir) / 'lfw_roc_curve.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ ROC curve saved: {roc_path}")
    plt.close()
    
    # Plota distribuição de similaridades
    plt.figure(figsize=(12, 6))
    
    # Pares positivos
    plt.subplot(1, 2, 1)
    pos_sims = similarities[labels == 1]
    plt.hist(pos_sims, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.3f}')
    plt.xlabel('Similarity Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Positive Pairs (Same Person)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Pares negativos
    plt.subplot(1, 2, 2)
    neg_sims = similarities[labels == 0]
    plt.hist(neg_sims, bins=50, color='red', alpha=0.7, edgecolor='black')
    plt.axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.3f}')
    plt.xlabel('Similarity Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Negative Pairs (Different People)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    dist_path = Path(output_dir) / 'lfw_similarity_distribution.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✅ Similarity distribution saved: {dist_path}")
    plt.close()
    
    return best_accuracy, best_threshold, auc_score, tar_at_far


# ========== AVALIAÇÃO ANTI-SPOOFING ==========

def evaluate_antispoofing(model, dataloader, device, threshold=0.5):
    """
    Avalia anti-spoofing no CASIA-FASD
    
    Returns:
        accuracy: acurácia de anti-spoofing
        predictions: probabilidades preditas
        labels: labels ground truth
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating anti-spoofing"):
            images = images.to(device)
            
            # Extrai spoofing score
            _, spoofing_logits = model.extract_features(
                images, 
                return_spoofing_score=True
            )
            
            # Converte para probabilidades
            spoofing_probs = torch.sigmoid(spoofing_logits).squeeze().cpu().numpy()
            
            all_probs.append(spoofing_probs)
            all_labels.append(labels.numpy())
    
    predictions = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    # Calcula accuracy
    binary_preds = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(labels, binary_preds)
    
    return accuracy, predictions, labels


def eval(model, device='cuda', lfw_root='data/val/lfw', casia_root=None, 
         batch_size=64, num_workers=4, spoof_threshold=0.5):
    """
    Função principal de avaliação
    
    Args:
        model: modelo treinado
        device: device (cuda ou cpu)
        lfw_root: path para LFW
        casia_root: path para CASIA-FASD (se None, não avalia anti-spoofing)
        batch_size: batch size
        num_workers: workers
        spoof_threshold: threshold para anti-spoofing
    
    Returns:
        lfw_accuracy: acurácia no LFW
        metrics: dicionário com todas as métricas
    """
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Transformações
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    metrics = {}
    
    # ========== LFW EVALUATION ==========
    print(f"\n[1/2] LFW Evaluation")
    print(f"LFW root: {lfw_root}")
    
    lfw_dataset = LFWDataset(root=lfw_root, transform=transform)
    lfw_dataloader = DataLoader(
        lfw_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    features1, features2, labels = extract_features(model, lfw_dataloader, device)
    similarities = compute_similarity(features1, features2)
    lfw_accuracy, threshold, auc_score, tar_at_far = compute_metrics(
        similarities, labels, output_dir='results/lfw'
    )
    
    metrics['lfw'] = {
        'accuracy': lfw_accuracy,
        'threshold': threshold,
        'auc': auc_score,
        'tar_at_far': tar_at_far
    }
    
    print(f"\nLFW Results:")
    print(f"  Accuracy:          {lfw_accuracy*100:.2f}%")
    print(f"  Best Threshold:    {threshold:.4f}")
    print(f"  AUC:               {auc_score:.4f}")
    
    # ========== ANTI-SPOOFING EVALUATION ==========
    if casia_root is not None:
        print(f"\n[2/2] Anti-Spoofing Evaluation (CASIA-FASD)")
        print(f"CASIA root: {casia_root}")
        print(f"Spoof threshold: {spoof_threshold}")
        
        # Avalia train + test
        spoof_metrics = {}
        
        for split in ['train', 'test']:
            print(f"\n  Evaluating {split} set...")
            
            try:
                casia_dataset = CASIAFASDDataset(
                    root=casia_root,
                    split=split,
                    transform=transform
                )
                
                casia_dataloader = DataLoader(
                    casia_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
                
                spoof_acc, preds, labs = evaluate_antispoofing(
                    model, casia_dataloader, device, threshold=spoof_threshold
                )
                
                spoof_metrics[split] = {
                    'accuracy': spoof_acc,
                    'threshold': spoof_threshold,
                    'num_samples': len(labs)
                }
                
                print(f"    Accuracy (SpfAcc): {spoof_acc*100:.2f}%")
                print(f"    Samples: {len(labs)}")
                
            except Exception as e:
                print(f"    ⚠️ Warning: Could not evaluate {split} set: {e}")
                spoof_metrics[split] = None
        
        metrics['antispoofing'] = spoof_metrics
    else:
        print(f"\n⚠️  Anti-spoofing evaluation skipped (no CASIA-FASD path provided)")
    
    # Resultados finais
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"LFW Accuracy:          {lfw_accuracy*100:.2f}%")
    
    if 'antispoofing' in metrics:
        for split, spoof_data in metrics['antispoofing'].items():
            if spoof_data is not None:
                print(f"SpfAcc ({split}):          {spoof_data['accuracy']*100:.2f}%")
    
    print("="*70 + "\n")
    
    return lfw_accuracy, metrics


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation (LFW + Anti-Spoofing)')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--lfw-root',
        type=str,
        default='data/val/lfw',
        help='Path to LFW dataset (default: data/val/lfw)'
    )
    parser.add_argument(
        '--casia-root',
        type=str,
        default=None,
        help='Path to CASIA-FASD (optional, for anti-spoofing evaluation)'
    )
    parser.add_argument(
        '--spoof-threshold',
        type=float,
        default=0.5,
        help='Threshold for anti-spoofing (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers (default: 4)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID (-1 for CPU, default: 0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"\n🖥️  Device: {device}")
    
    # Carrega modelo
    print(f"\n📂 Loading model from: {args.checkpoint}")
    model = mobilenetv3_large_multitask(embedding_dim=512).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"✓ Model from epoch {checkpoint['epoch']}")
    if 'num_classes' in checkpoint:
        print(f"✓ Trained with {checkpoint['num_classes']:,} classes")
    
    # Avalia
    lfw_accuracy, metrics = eval(
        model=model,
        device=device,
        lfw_root=args.lfw_root,
        casia_root=args.casia_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spoof_threshold=args.spoof_threshold
    )
    
    # Salva métricas
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = Path(args.output_dir) / 'evaluation_metrics.json'
    metrics_clean = convert_to_python_types(metrics)
    with open(metrics_file, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    
    print(f"✅ Metrics saved: {metrics_file}\n")


if __name__ == '__main__':
    main()