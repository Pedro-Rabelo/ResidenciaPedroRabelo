# Face Recognition Pipeline with Multi-Task Learning

Pipeline completo para reconhecimento facial usando MobileNetV3 com aprendizado multi-tarefa: classificação de identidades, regressão de landmarks faciais e detecção de liveness (anti-spoofing).

## Features

- **Backbone**: MobileNetV3-Large otimizado para faces
- **Multi-Task Learning**:
  - Classificação de identidades (CosFace/ArcFace)
  - Regressão de 5 landmarks faciais
  - Detecção de liveness (anti-spoofing)
- **Detecção facial**: MTCNN com alinhamento automático
- **Datasets**: VGGFace2 + CASIA-FASD (opcional)
- **Validação**: LFW benchmark

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Crie um subset do VGGFace2 (opcional):

```bash
python create_subset.py \
    --input-root <caminho_dataset_original> \
    --output-root <caminho_saida_subset> \
    --n-identities <numero_identidades>
```

### 2. Preprocess Images

Detecta faces, alinha e extrai landmarks:

```bash
python preprocess_with_landmarks_BATCH.py \
    --input-root <caminho_imagens_originais> \
    --output-root <caminho_imagens_alinhadas> \
    --landmarks-json <caminho_arquivo_landmarks.json> \
    --gpu <gpu_id>
```

### 3. Train Model

**Sem anti-spoofing:**

```bash
python train_multitask.py \
    --root <caminho_imagens_alinhadas> \
    --landmarks-json <caminho_landmarks.json> \
    --batch-size <tamanho_batch> \
    --epochs <numero_epochs> \
    --lfw-root <caminho_lfw>
```

**Com anti-spoofing (requer CASIA-FASD):**

```bash
python train_multitask.py \
    --root <caminho_imagens_alinhadas> \
    --landmarks-json <caminho_landmarks.json> \
    --casia-root <caminho_casia_fasd> \
    --use-hybrid-dataset \
    --casia-ratio <ratio> \
    --spoofing-weight <peso> \
    --batch-size <tamanho_batch> \
    --epochs <numero_epochs> \
    --lfw-root <caminho_lfw>
```

### 4. Inference

**Comparar duas faces:**

```bash
python inference_multitask.py \
    --mode pair \
    --checkpoint <caminho_checkpoint.ckpt> \
    --img1 <caminho_imagem1> \
    --img2 <caminho_imagem2> \
    --similarity-threshold <threshold> \
    --spoof-threshold <threshold>
```

**Testar liveness:**

```bash
python inference_multitask.py \
    --mode single \
    --checkpoint <caminho_checkpoint.ckpt> \
    --img1 <caminho_imagem> \
    --spoof-threshold <threshold>
```

**Extrair embeddings em lote:**

```bash
python inference_multitask.py \
    --mode batch \
    --checkpoint <caminho_checkpoint.ckpt> \
    --folder <caminho_pasta_imagens> \
    --output <caminho_saida.json>
```

### 5. Evaluation

```bash
python evaluate.py \
    --checkpoint <caminho_checkpoint.ckpt> \
    --lfw-root <caminho_lfw> \
    --casia-root <caminho_casia_fasd> \
    --batch-size <tamanho_batch>
```

## Dataset Structure

### VGGFace2

```
data/
├── raw/
│   └── vggface2_112x112/
│       ├── n000001/
│       │   ├── 0001_01.jpg
│       │   └── ...
│       └── ...
└── train/
    ├── vggface2_aligned_112x112/
    └── vggface2_landmarks.json
```

### LFW (Validation)

```
data/val/
├── lfw/
│   ├── Aaron_Eckhart/
│   │   └── Aaron_Eckhart_0001.jpg
│   └── ...
└── pairs.txt
```

### CASIA-FASD (Anti-Spoofing)

```
data/casia-fasd/
├── train/
│   ├── live/
│   └── spoof/
└── test/
    ├── live/
    └── spoof/
```

## Model Architecture

```
MobileNetV3-Large Backbone
├── Feature Extractor (features)
├── Embedding Head (GDC + Linear) → 512D embeddings
├── Landmark Head → 10 values (5 landmarks × 2)
└── Spoofing Head → 1 value (logit)
```

## Loss Function

Combinação ponderada de três objetivos:
- Classification: CosFace (Margin Cosine Product)
- Landmarks: SmoothL1 Loss ou Wing Loss
- Anti-Spoofing: BCEWithLogitsLoss

## Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | VGGFace2 aligned images path | Required |
| `--landmarks-json` | Landmarks JSON file | Required |
| `--casia-root` | CASIA-FASD path (enables anti-spoofing) | None |
| `--use-hybrid-dataset` | Use VGGFace2 + CASIA hybrid dataset | False |
| `--casia-ratio` | CASIA samples ratio | 0.3 |
| `--batch-size` | Batch size | 256 |
| `--epochs` | Number of epochs | 30 |
| `--lr` | Initial learning rate | 0.1 |
| `--landmark-weight` | Landmark loss weight | 0.5 |
| `--spoofing-weight` | Anti-spoofing loss weight | 0.3 |
| `--use-wing-loss` | Use Wing Loss for landmarks | False |
| `--lfw-root` | LFW dataset path | data/val |

## Inference Modes

| Mode | Description | Required Args |
|------|-------------|---------------|
| `single` | Test single image for liveness | `--img1` |
| `pair` | Compare two faces + spoofing check | `--img1`, `--img2` |
| `compare` | Compare two faces (legacy) | `--img1`, `--img2` |
| `extract` | Extract embedding from single image | `--img1` |
| `batch` | Extract embeddings from folder | `--folder` |
| `test-spoof` | Test only anti-spoofing | `--img1` |

## Checkpoint Format

```python
{
    'epoch': int,
    'model': OrderedDict,                # Model state dict
    'classification_head': OrderedDict,  # MCP head state dict
    'optimizer': dict,
    'lr_scheduler': dict,
    'best_lfw_accuracy': float,
    'num_classes': int,
    'args': Namespace
}
```

## Landmarks Format

JSON file with normalized landmarks (0-1 range):

```json
{
    "identity/image.jpg": [
        x1, y1,  // Left Eye
        x2, y2,  // Right Eye
        x3, y3,  // Nose
        x4, y4,  // Mouth Left
        x5, y5   // Mouth Right
    ]
}
```

## Evaluation Metrics

### Recognition (LFW)
- Accuracy
- ROC curve (AUC)
- TAR @ FAR (0.1%, 1%, 10%)

### Anti-Spoofing (CASIA-FASD)
- Accuracy (SpfAcc)
- Per-split evaluation (train/test)

## Project Structure

```
Pipeline/
├── models/
│   ├── mobilenetv3.py              # MobileNetV3 backbone
│   └── mobilenetv3_multitask.py    # Multi-task wrapper
├── utils/
│   ├── dataset_landmarks.py        # Dataset loaders
│   ├── metrics.py                  # CosFace, ArcFace
│   ├── multitask_loss.py           # Multi-task loss functions
│   ├── layers.py                   # Custom layers (GDC, SE)
│   ├── general.py                  # Training utilities
│   └── spoofing_dataset.py         # CASIA-FASD loader
├── preprocess_with_landmarks_BATCH.py
├── train_multitask.py
├── inference_multitask.py
├── evaluate.py
├── create_subset.py
└── README.md
```

## Citation

Se usar este código, considere citar:

```bibtex
@misc{face-recognition-multitask,
  title={Face Recognition Pipeline with Multi-Task Learning},
  author={Pedro Rabelo Mendonça},
  year={2025}
}
```

## License

MIT License
