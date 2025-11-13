import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset_landmarks import HybridDataset

print("="*60)
print("TESTE RÁPIDO DO DATASET")
print("="*60)

# Transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Tenta carregar dataset
try:
    print("\nCarregando HybridDataset...")
    dataset = HybridDataset(
        vggface_root='data/train/vggface2_aligned_200',
        landmarks_json='data/train/vggface2_landmarks_200.json',
        casia_root='data/casia-fasd',
        transform=transform,
        casia_ratio=0.3
    )
    
    print(f"\n✅ Dataset carregado com sucesso!")
    print(f"Total samples: {len(dataset)}")
    
    # Testa DataLoader
    print("\nTestando DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Pega um batch
    for images, targets, landmarks, is_spoof in loader:
        print(f"\n✅ Batch carregado com sucesso!")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets: {targets}")
        print(f"  Landmarks shape: {landmarks.shape}")
        print(f"  Is spoof: {is_spoof}")
        print(f"\n  Estatísticas:")
        print(f"    - VGGFace2 samples (target >= 0): {(targets >= 0).sum().item()}")
        print(f"    - CASIA samples (target == -1): {(targets == -1).sum().item()}")
        print(f"    - Real faces (spoof == 0): {(is_spoof == 0).sum().item()}")
        print(f"    - Fake faces (spoof == 1): {(is_spoof == 1).sum().item()}")
        break
    
    print("\n" + "="*60)
    print("✅ TUDO OK! PRONTO PARA TREINAR!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    print("\nVerifique:")
    print("  1. Caminhos dos datasets")
    print("  2. Formato do landmarks JSON")
    print("  3. Estrutura das pastas CASIA-FASD")
    import traceback
    traceback.print_exc()