import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CASIAFASDDataset(Dataset):
    """
    Dataset para CASIA-FASD Anti-Spoofing
    
    Estrutura esperada:
    casia-fasd/
    ├── train/
    │   ├── live/     (imagens reais, label=0)
    │   └── spoof/    (imagens fake, label=1)
    └── test/
        ├── live/
        └── spoof/
    """
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root: Caminho para casia-fasd/
            split: 'train' ou 'test'
            transform: Transformações da imagem
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Monta caminhos
        split_path = self.root / split
        live_path = split_path / 'live'
        spoof_path = split_path / 'spoof'
        
        # Valida estrutura
        if not split_path.exists():
            raise ValueError(f"Split path não existe: {split_path}")
        if not live_path.exists() or not spoof_path.exists():
            raise ValueError(f"Pastas live/spoof não encontradas em {split_path}")
        
        # Carrega samples
        self.samples = []
        
        # Imagens LIVE (label=0)
        for img_path in self._get_images(live_path):
            self.samples.append({
                'path': str(img_path),
                'label': 0,  # Real
                'type': 'live'
            })
        
        # Imagens SPOOF (label=1)
        for img_path in self._get_images(spoof_path):
            self.samples.append({
                'path': str(img_path),
                'label': 1,  # Fake
                'type': 'spoof'
            })
        
        print(f"CASIA-FASD {split}: {len(self.samples)} samples")
        print(f"  Live: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"  Spoof: {sum(1 for s in self.samples if s['label'] == 1)}")
    
    def _get_images(self, directory):
        """Retorna lista de caminhos de imagens"""
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"**/*{ext}"))  # Busca recursiva
        return sorted(images)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Carrega imagem
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Label de spoofing (0=real, 1=fake)
        label = sample['label']
        
        return img, label
    
    def get_sample_weights(self):
        """
        Retorna pesos para balanceamento de classes
        Útil para WeightedRandomSampler
        """
        labels = [s['label'] for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        class_weights = [1.0 / count for count in class_counts]
        
        sample_weights = [class_weights[label] for label in labels]
        return sample_weights


class VGGFace2WithSpoofing(Dataset):
    """
    Dataset híbrido: VGGFace2 (identidades) + CASIA-FASD (spoofing)
    
    Retorna: (image, identity_label, landmarks, is_spoof)
    """
    
    def __init__(self, vggface_root, landmarks_json, casia_root=None, 
                 transform=None, mix_ratio=0.3):
        """
        Args:
            vggface_root: Caminho para VGGFace2 alinhado
            landmarks_json: JSON com landmarks
            casia_root: Caminho para CASIA-FASD (opcional)
            transform: Transformações
            mix_ratio: Proporção de samples CASIA no batch (0.0 a 1.0)
        """
        import json
        
        self.transform = transform
        self.mix_ratio = mix_ratio
        
        # Carrega VGGFace2 (todas são LIVE)
        self.vggface_samples = []
        self.class_to_idx = {}
        self.classes = []
        
        with open(landmarks_json, 'r') as f:
            landmarks_data = json.load(f)
        
        vggface_path = Path(vggface_root)
        class_dirs = sorted([d for d in vggface_path.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
            
            for img_file in class_dir.glob("*.jpg"):
                key = f"{class_name}/{img_file.name}"
                if key in landmarks_data:
                    self.vggface_samples.append({
                        'path': str(img_file),
                        'identity': idx,
                        'landmarks': landmarks_data[key],
                        'is_spoof': 0  # VGGFace2 = sempre real
                    })
        
        print(f"VGGFace2: {len(self.vggface_samples)} samples, {len(self.classes)} identities")
        
        # Carrega CASIA-FASD (sem identidade, só spoofing)
        self.casia_samples = []
        if casia_root:
            for split in ['train']:  # Só usa train do CASIA
                live_path = Path(casia_root) / split / 'live'
                spoof_path = Path(casia_root) / split / 'spoof'
                
                # Live samples (label identidade = -1, is_spoof=0)
                for img_path in self._get_images(live_path):
                    self.casia_samples.append({
                        'path': str(img_path),
                        'identity': -1,  # Sem identidade
                        'landmarks': [0.0] * 10,  # Dummy landmarks
                        'is_spoof': 0
                    })
                
                # Spoof samples (label identidade = -1, is_spoof=1)
                for img_path in self._get_images(spoof_path):
                    self.casia_samples.append({
                        'path': str(img_path),
                        'identity': -1,
                        'landmarks': [0.0] * 10,
                        'is_spoof': 1
                    })
            
            print(f"CASIA-FASD: {len(self.casia_samples)} samples")
        
        # Combina datasets
        self.all_samples = self.vggface_samples + self.casia_samples
        print(f"Total: {len(self.all_samples)} samples\n")
    
    def _get_images(self, directory):
        extensions = ['.jpg', '.jpeg', '.png']
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
        return sorted(images)
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        
        # Carrega imagem
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Retorna 4 valores
        identity = sample['identity']
        landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
        is_spoof = sample['is_spoof']
        
        return img, identity, landmarks, is_spoof
    
    def get_num_classes(self):
        """Retorna número de identidades (só VGGFace2)"""
        return len(self.classes)