import torch
import torch.nn as nn
from models.mobilenetv3 import mobilenet_v3_large

class MobileNetV3MultiTask(nn.Module):
    """
    MobileNetV3 com TRÊS cabeças:
    1. Embedding head (reconhecimento facial)
    2. Landmark head (regressão de landmarks)
    3. Spoofing head (detecção de liveness) ← NOVO
    """
    
    def __init__(self, embedding_dim=512, num_landmarks=10):
        super().__init__()
        
        # Backbone MobileNetV3
        base_model = mobilenet_v3_large(embedding_dim=embedding_dim)
        
        # Usa features do MobileNetV3
        self.features = base_model.features
        
        # GDC layer original (para embedding)
        self.gdc = base_model.output_layer
        
        # Cabeça de landmarks (branch auxiliar)
        lastconv_channels = 960  # MobileNetV3-Large antes do GDC
        
        self.landmark_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lastconv_channels, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks)  # 10 valores: (x,y) * 5 landmarks
        )
        
        # ========== NOVO: Cabeça de Anti-Spoofing ==========
        self.spoofing_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lastconv_channels, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.5),  # Dropout maior para regularização
            nn.Linear(256, 1)  # Saída binária (logit, antes de sigmoid)
        )
        
        # Inicialização da cabeça de landmarks
        for m in self.landmark_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Inicialização da cabeça de spoofing
        for m in self.spoofing_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_landmarks=True, return_spoofing=True):
        """
        Args:
            x: imagem [B, 3, 112, 112]
            return_landmarks: se True, retorna landmarks preditos
            return_spoofing: se True, retorna score anti-spoofing
        
        Returns:
            embedding: [B, 512]
            landmarks: [B, 10] (se return_landmarks=True)
            spoofing_logit: [B, 1] (se return_spoofing=True)
        """
        # Features do backbone
        features = self.features(x)  # [B, 960, 7, 7]
        
        # Embedding para reconhecimento
        embedding = self.gdc(features)  # [B, 512]
        
        outputs = [embedding]
        
        if return_landmarks:
            landmarks = self.landmark_head(features)  # [B, 10]
            outputs.append(landmarks)
        
        if return_spoofing:
            spoofing_logit = self.spoofing_head(features)  # [B, 1]
            outputs.append(spoofing_logit)
        
        # Retorna tupla com os outputs necessários
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def extract_features(self, x, return_spoofing_score=False):
        """
        Para inferência: retorna embedding e opcionalmente score anti-spoofing
        
        Args:
            x: imagem [B, 3, 112, 112]
            return_spoofing_score: se True, retorna (embedding, spoof_prob)
        
        Returns:
            embedding: [B, 512]
            spoof_prob: [B, 1] probabilidade de ser spoof (0-1) (opcional)
        """
        if return_spoofing_score:
            embedding, _, spoofing_logit = self.forward(
                x, 
                return_landmarks=False, 
                return_spoofing=True
            )
            spoof_prob = torch.sigmoid(spoofing_logit)  # Converte logit para probabilidade
            return embedding, spoof_prob
        else:
            return self.forward(x, return_landmarks=False, return_spoofing=False)


def mobilenetv3_large_multitask(embedding_dim=512, **kwargs):
    """Factory function"""
    return MobileNetV3MultiTask(embedding_dim=embedding_dim, **kwargs)