import os
import json
import argparse
from pathlib import Path

def prepare_casia_metadata(casia_root, output_json):
    """
    Cria JSON com metadados do CASIA-FASD
    
    Formato:
    {
        "train/live/image.jpg": {"is_spoof": 0, "split": "train"},
        "train/spoof/image.jpg": {"is_spoof": 1, "split": "train"},
        ...
    }
    """
    metadata = {}
    casia_path = Path(casia_root)
    
    for split in ['train', 'test']:
        split_path = casia_path / split
        
        if not split_path.exists():
            print(f"⚠️  Split {split} não encontrado em {casia_root}")
            continue
        
        # Live images (label=0)
        live_path = split_path / 'live'
        if live_path.exists():
            for img in live_path.glob("*.jpg"):
                key = f"{split}/live/{img.name}"
                metadata[key] = {
                    "is_spoof": 0,
                    "split": split,
                    "type": "live"
                }
        
        # Spoof images (label=1)
        spoof_path = split_path / 'spoof'
        if spoof_path.exists():
            for img in spoof_path.glob("*.jpg"):
                key = f"{split}/spoof/{img.name}"
                metadata[key] = {
                    "is_spoof": 1,
                    "split": split,
                    "type": "spoof"
                }
    
    # Salva JSON
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Estatísticas
    train_live = sum(1 for v in metadata.values() if v['split'] == 'train' and v['is_spoof'] == 0)
    train_spoof = sum(1 for v in metadata.values() if v['split'] == 'train' and v['is_spoof'] == 1)
    test_live = sum(1 for v in metadata.values() if v['split'] == 'test' and v['is_spoof'] == 0)
    test_spoof = sum(1 for v in metadata.values() if v['split'] == 'test' and v['is_spoof'] == 1)
    
    print(f"\n{'='*60}")
    print("CASIA-FASD METADATA CRIADO")
    print(f"{'='*60}")
    print(f"Train: {train_live + train_spoof} samples")
    print(f"  Live:  {train_live}")
    print(f"  Spoof: {train_spoof}")
    print(f"\nTest: {test_live + test_spoof} samples")
    print(f"  Live:  {test_live}")
    print(f"  Spoof: {test_spoof}")
    print(f"\nTotal: {len(metadata)} samples")
    print(f"Salvo em: {output_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara metadados CASIA-FASD")
    parser.add_argument(
        '--casia-root',
        type=str,
        required=True,
        help='Caminho para casia-fasd/ (contém train/ e test/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/casia_fasd_metadata.json',
        help='Caminho para salvar JSON'
    )
    
    args = parser.parse_args()
    
    prepare_casia_metadata(args.casia_root, args.output)