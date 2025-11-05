import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    Loss combinada para multi-task learning:
    1. Classification loss (cross-entropy via MCP)
    2. Landmark regression loss (auxiliar)
    3. Anti-spoofing loss (binária) ← NOVO
    """
    
    def __init__(self, landmark_weight=0.5, spoofing_weight=0.3):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.spoofing_weight = spoofing_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.landmark_loss = nn.SmoothL1Loss()
        
        # ========== NOVO: Loss de Anti-Spoofing ==========
        # BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid
        # Mais estável numericamente que BCE(sigmoid(x))
        self.spoofing_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, cls_output, landmarks_pred, spoofing_pred, 
                targets, landmarks_gt, spoofing_gt):
        """
        Args:
            cls_output: saída do MCP [B, num_classes]
            landmarks_pred: landmarks preditos [B, 10]
            spoofing_pred: logits anti-spoofing [B, 1] ← NOVO
            targets: labels de identidade [B]
            landmarks_gt: landmarks ground truth [B, 10]
            spoofing_gt: labels de spoofing [B] (0=real, 1=fake) ← NOVO
        
        Returns:
            total_loss: loss combinada
            loss_dict: dicionário com losses individuais
        """
        # Loss de classificação (identidade)
        L_cls = self.classification_loss(cls_output, targets)
        
        # Loss de landmarks (regressão)
        L_landmark = self.landmark_loss(landmarks_pred, landmarks_gt)
        
        # ========== NOVO: Loss de Anti-Spoofing ==========
        # Converte spoofing_gt para float e ajusta dimensão
        spoofing_gt_float = spoofing_gt.float().unsqueeze(1)  # [B] → [B, 1]
        L_spoofing = self.spoofing_loss(spoofing_pred, spoofing_gt_float)
        
        # Loss total ponderada
        total_loss = (L_cls + 
                     self.landmark_weight * L_landmark + 
                     self.spoofing_weight * L_spoofing)
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': L_cls.item(),
            'landmark': L_landmark.item(),
            'spoofing': L_spoofing.item()  # NOVO
        }
        
        return total_loss, loss_dict


class WingLoss(nn.Module):
    """
    Wing Loss para landmarks (melhor que L1/L2)
    Referência: Wing Loss for Robust Facial Landmark Localisation (CVPR 2018)
    """
    
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    
    def forward(self, pred, target):
        delta = (target - pred).abs()
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        return loss.mean()


class MultiTaskLossAdvanced(nn.Module):
    """
    Loss avançada com Wing Loss e Anti-Spoofing
    """
    
    def __init__(self, landmark_weight=0.5, spoofing_weight=0.3, use_wing_loss=True):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.spoofing_weight = spoofing_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        
        if use_wing_loss:
            self.landmark_loss = WingLoss(omega=10, epsilon=2)
        else:
            self.landmark_loss = nn.SmoothL1Loss()
        
        # Anti-spoofing loss
        self.spoofing_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, cls_output, landmarks_pred, spoofing_pred,
                targets, landmarks_gt, spoofing_gt):
        # Loss de classificação
        L_cls = self.classification_loss(cls_output, targets)
        
        # Loss de landmarks
        L_landmark = self.landmark_loss(landmarks_pred, landmarks_gt)
        
        # Loss de anti-spoofing
        spoofing_gt_float = spoofing_gt.float().unsqueeze(1)
        L_spoofing = self.spoofing_loss(spoofing_pred, spoofing_gt_float)
        
        # Loss total
        total_loss = (L_cls + 
                     self.landmark_weight * L_landmark +
                     self.spoofing_weight * L_spoofing)
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': L_cls.item(),
            'landmark': L_landmark.item(),
            'spoofing': L_spoofing.item()
        }
        
        return total_loss, loss_dict