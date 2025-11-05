import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from pathlib import Path

class ImageFolderWithLandmarks(Dataset):
    """
    Dataset que carrega:
    - Imagem alinhada
    - Label da identidade
    - Landmarks normalizados
    - Label de spoofing (NOVO) - padrão 0 (real) se não fornecido
    """
    
    def __init__(self, root, landmarks_json, spoofing_json=None, transform=None):
        """
        Args:
            root: Diretório com imagens alinhadas
            landmarks_json: JSON com landmarks
            spoofing_json: JSON com labels de spoofing (opcional)
            transform: Transformações de imagem
        """
        self.root = root
        self.transform = transform
        
        # Carrega landmarks
        with open(landmarks_json, 'r') as f:
            self.landmarks_data = json.load(f)
        
        # ========== NOVO: Carrega labels de spoofing (se fornecido) ==========
        self.spoofing_data = {}
        if spoofing_json and os.path.exists(spoofing_json):
            with open(spoofing_json, 'r') as f:
                self.spoofing_data = json.load(f)
            print(f"✓ Loaded spoofing labels from {spoofing_json}")
        else:
            print(f"⚠️  No spoofing labels provided, assuming all images are REAL (label=0)")
        
        # Constrói lista de samples
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        root_path = Path(root)
        class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
            
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))
            
            for img_file in image_files:
                key = f"{class_name}/{img_file.name}"
                
                # Apenas adiciona se tiver landmarks
                if key in self.landmarks_data:
                    # ========== NOVO: Busca label de spoofing ==========
                    # Padrão: 0 (real) se não estiver no JSON
                    is_spoof = self.spoofing_data.get(key, 0)
                    
                    self.samples.append({
                        'path': str(img_file),
                        'label': idx,
                        'landmarks': self.landmarks_data[key],
                        'is_spoof': is_spoof,  # NOVO
                        'key': key
                    })
        
        # ========== NOVO: Estatísticas de spoofing ==========
        num_real = sum(1 for s in self.samples if s['is_spoof'] == 0)
        num_fake = sum(1 for s in self.samples if s['is_spoof'] == 1)
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"  Real images: {num_real}")
        print(f"  Fake images: {num_fake}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Carrega imagem
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Landmarks como tensor
        landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
        
        label = sample['label']
        is_spoof = sample['is_spoof']  # NOVO
        
        # ========== NOVO: Retorna 4 valores ==========
        return img, label, landmarks, is_spoof
    
    def get_num_classes(self):
        """Retorna número de identidades"""
        return len(self.classes)


class HybridDataset(Dataset):
    """
    Dataset híbrido que combina VGGFace2 + CASIA-FASD
    
    Estratégia:
    - VGGFace2: Todas as 3 tasks (identidade + landmarks + spoofing=0)
    - CASIA-FASD: Apenas spoofing (identidade=-1, landmarks=dummy)
    """
    
    def __init__(self, vggface_root, landmarks_json, casia_root, 
                 transform=None, casia_ratio=0.3):
        """
        Args:
            vggface_root: Caminho para VGGFace2 alinhado
            landmarks_json: JSON com landmarks do VGGFace2
            casia_root: Caminho para CASIA-FASD
            transform: Transformações
            casia_ratio: Proporção de amostras CASIA no dataset (0.0 a 1.0)
        """
        self.transform = transform
        
        # ========== Carrega VGGFace2 ==========
        print("\n" + "="*60)
        print("CARREGANDO VGGFACE2")
        print("="*60)
        
        with open(landmarks_json, 'r') as f:
            landmarks_data = json.load(f)
        
        self.vggface_samples = []
        self.class_to_idx = {}
        self.classes = []
        
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
                        'is_spoof': 0,  # VGGFace2 = sempre real
                        'source': 'vggface2'
                    })
        
        print(f"✓ VGGFace2: {len(self.vggface_samples)} samples")
        print(f"✓ Identidades: {len(self.classes)}")
        
        # ========== Carrega CASIA-FASD ==========
        print("\n" + "="*60)
        print("CARREGANDO CASIA-FASD")
        print("="*60)
        
        self.casia_samples = []
        casia_path = Path(casia_root)
        
        for split in ['train']:  # Apenas train do CASIA
            split_path = casia_path / split
            
            if not split_path.exists():
                print(f"⚠️  Warning: {split_path} não encontrado")
                continue
            
            # Live samples
            live_path = split_path / 'live'
            if live_path.exists():
                for img_file in self._get_images(live_path):
                    self.casia_samples.append({
                        'path': str(img_file),
                        'identity': -1,  # Sem identidade conhecida
                        'landmarks': [0.0] * 10,  # Dummy landmarks
                        'is_spoof': 0,  # Real
                        'source': 'casia_live'
                    })
            
            # Spoof samples
            spoof_path = split_path / 'spoof'
            if spoof_path.exists():
                for img_file in self._get_images(spoof_path):
                    self.casia_samples.append({
                        'path': str(img_file),
                        'identity': -1,
                        'landmarks': [0.0] * 10,
                        'is_spoof': 1,  # Fake
                        'source': 'casia_spoof'
                    })
        
        casia_live = sum(1 for s in self.casia_samples if s['is_spoof'] == 0)
        casia_spoof = sum(1 for s in self.casia_samples if s['is_spoof'] == 1)
        
        print(f"✓ CASIA-FASD: {len(self.casia_samples)} samples")
        print(f"  Live:  {casia_live}")
        print(f"  Spoof: {casia_spoof}")
        
        # ========== Combina datasets com ratio ==========
        # Calcula quantas amostras CASIA usar
        num_casia_to_use = int(len(self.vggface_samples) * casia_ratio)
        
        if num_casia_to_use > len(self.casia_samples):
            print(f"\n⚠️  Warning: Requested {num_casia_to_use} CASIA samples, "
                  f"but only {len(self.casia_samples)} available")
            num_casia_to_use = len(self.casia_samples)
        
        # Amostragem balanceada de CASIA
        import random
        random.seed(42)
        casia_subset = random.sample(self.casia_samples, num_casia_to_use)
        
        self.all_samples = self.vggface_samples + casia_subset
        
        print("\n" + "="*60)
        print("DATASET HÍBRIDO CRIADO")
        print("="*60)
        print(f"Total: {len(self.all_samples)} samples")
        print(f"  VGGFace2: {len(self.vggface_samples)}")
        print(f"  CASIA-FASD: {len(casia_subset)}")
        print(f"Ratio CASIA: {len(casia_subset)/len(self.all_samples):.2%}")
        
        # Estatísticas finais de spoofing
        total_real = sum(1 for s in self.all_samples if s['is_spoof'] == 0)
        total_fake = sum(1 for s in self.all_samples if s['is_spoof'] == 1)
        print(f"\nDistribuição Spoofing:")
        print(f"  Real: {total_real} ({total_real/len(self.all_samples):.1%})")
        print(f"  Fake: {total_fake} ({total_fake/len(self.all_samples):.1%})")
        print("="*60 + "\n")
    
    def _get_images(self, directory):
        """Lista imagens em um diretório"""
        extensions = ['.jpg', '.jpeg', '.png']
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"**/*{ext}"))
        return sorted(images)
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        
        # Carrega imagem
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        identity = sample['identity']
        landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
        is_spoof = sample['is_spoof']
        
        return img, identity, landmarks, is_spoof
    
    def get_num_classes(self):
        """Retorna número de identidades (só VGGFace2)"""
        return len(self.classes)


def create_validation_split_with_landmarks(dataset, val_split=0.1, seed=42):
    """
    Split estratificado considerando landmarks E spoofing
    """
    from torch.utils.data import Subset
    import random
    
    random.seed(seed)
    
    # Organiza por classe (identidade)
    class_indices = {}
    for idx, sample in enumerate(dataset.samples):
        label = sample['label']
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # Split estratificado por identidade
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_split))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset