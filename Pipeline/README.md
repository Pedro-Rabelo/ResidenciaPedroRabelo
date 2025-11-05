# Pipeline de Reconhecimento Facial com Multi-Task Learning

## Visão Geral

Pipeline completo para treinamento e inferência de modelos de reconhecimento facial baseado em MobileNetV3 com aprendizado multi-tarefa (Multi-Task Learning). O sistema implementa uma arquitetura que combina classificação de identidades faciais com regressão de landmarks e detecção de liveness (anti-spoofing) como tarefas auxiliares, melhorando a robustez e qualidade dos embeddings gerados.

### Características Principais

- Backbone MobileNetV3-Large otimizado para reconhecimento facial
- Aprendizado multi-tarefa com classificação, regressão de landmarks e anti-spoofing
- Anti-Spoofing integrada ao modelo
- Suporte a dataset CASIA-FASD para treinamento anti-spoofing
- Dataset híbrido (VGGFace2 + CASIA-FASD) para treinamento multi-objetivo
- Detecção e alinhamento facial automatizado com MTCNN
- Validação em benchmark LFW (Labeled Faces in the Wild)
- Suporte a treinamento em VGGFace2 dataset
- Inferência otimizada para CPU e GPU

## Arquitetura

### Modelo Multi-Task

O modelo utiliza MobileNetV3-Large como backbone com três cabeças especializadas:

**Embedding Head**
- Extração de features para reconhecimento facial
- Dimensão de embedding: 512D
- Global Depthwise Convolution (GDC) layer

**Landmark Head**
- Predição de 5 landmarks faciais (olhos, nariz, cantos da boca)
- Saída: 10 valores normalizados (x, y para cada landmark)
- Arquitetura auxiliar para melhorar representações faciais

**Spoofing Head**
- Detecção de ataques de apresentação
- Saída: 1 valor (probabilidade de ser fake, 0-1)
- Classificação binária: real vs. fake

### Função de Perda

A loss total combina três objetivos:

```
L_total = L_classification + λ₁ × L_landmark + λ₂ × L_spoofing
```

- **L_classification**: Margin Cosine Product (CosFace) para classificação de identidades
- **L_landmark**: SmoothL1 Loss ou Wing Loss para regressão de landmarks
- **L_spoofing**: BCEWithLogitsLoss para detecção de liveness
- **λ₁**: Peso da tarefa de landmarks (padrão: 0.5)
- **λ₂**: Peso da tarefa anti-spoofing (padrão: 0.3)

## Instalação

### Requisitos

```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.19.0
tqdm>=4.60.0
facenet-pytorch  # Para MTCNN
```

### Instalação de Dependências

```bash
pip install torch torchvision opencv-python pillow numpy tqdm
pip install facenet-pytorch
```

## Dataset CASIA-FASD (Opcional)

Para habilitar anti-spoofing, baixe o CASIA Face Anti-Spoofing Database.

**Estrutura esperada:**
```
data/
├── casia-fasd/
│   ├── train/
│   │   ├── live/     # Imagens reais
│   │   └── spoof/    # Imagens fake (fotos, vídeos, máscaras)
│   └── test/
│       ├── live/
│       └── spoof/
```

**Preparar metadados:**
```bash
python prepare_spoofing_labels.py \
    --casia-root data/casia-fasd \
    --output data/casia_fasd_metadata.json
```

## Uso

### 1. Pré-processamento do Dataset

O script de pré-processamento realiza detecção facial com MTCNN, validação, alinhamento e extração de landmarks.

```bash
python preprocess_with_landmarks_BATCH.py \
    --input-root data/raw/vggface2_112x112 \
    --output-root data/train/vggface2_aligned_112x112 \
    --landmarks-json data/train/vggface2_landmarks.json \
    --min-size 20 \
    --min-images-per-class 2
```

**Parâmetros:**
- `--input-root`: Diretório com imagens originais do VGGFace2
- `--output-root`: Diretório de saída para imagens alinhadas (112x112)
- `--landmarks-json`: Arquivo JSON para armazenar landmarks normalizados
- `--min-size`: Tamanho mínimo da face em pixels (padrão: 20)
- `--min-images-per-class`: Mínimo de imagens por identidade (padrão: 2)
- `--batch-size`: Tamanho do batch para GPU (padrão: 32)
- `--gpu`: ID da GPU (-1 para CPU, padrão: 0)

**Saída:**
- Imagens alinhadas em 112x112 pixels usando MTCNN
- Arquivo JSON com landmarks normalizados
- Estatísticas de processamento

### 2. Treinamento

#### Treinamento Padrão (sem anti-spoofing)

```bash
python train_multitask.py \
    --root data/train/vggface2_aligned_112x112 \
    --landmarks-json data/train/vggface2_landmarks.json \
    --batch-size 256 \
    --epochs 30 \
    --lr 0.1 \
    --landmark-weight 0.5 \
    --save-path weights/vggface2 \
    --lfw-root data/val
```

#### Treinamento com Anti-Spoofing

**Opção: Dataset Híbrido (VGGFace2 + CASIA-FASD)**
```bash
python train_multitask.py \
    --root data/train/vggface2_aligned_112x112 \
    --landmarks-json data/train/vggface2_landmarks.json \
    --casia-root data/casia-fasd \
    --use-hybrid-dataset \
    --casia-ratio 0.3 \
    --spoofing-weight 0.3 \
    --batch-size 256 \
    --epochs 30 \
    --lr 0.1 \
    --landmark-weight 0.5 \
    --save-path weights/vggface2 \
    --lfw-root data/val
```

**Parâmetros de Dataset:**
- `--root`: Diretório com imagens alinhadas
- `--landmarks-json`: Arquivo JSON com landmarks
- `--train-split`: Proporção treino/validação (padrão: 0.8)
- `--min-images-per-class`: Mínimo de imagens por identidade (padrão: 2)

**Parâmetros Anti-Spoofing:**
- `--casia-root`: Diretório do CASIA-FASD (habilita anti-spoofing)
- `--use-hybrid-dataset`: Ativa dataset híbrido VGGFace2 + CASIA
- `--casia-ratio`: Proporção de amostras CASIA (padrão: 0.3 = 30%)
- `--spoofing-weight`: Peso da loss anti-spoofing (padrão: 0.3)
- `--disable-spoofing-check`: Desativa verificação de spoofing

**Parâmetros de Modelo:**
- `--embedding-dim`: Dimensão do embedding (padrão: 512)
- `--landmark-weight`: Peso da loss de landmarks (padrão: 0.5)
- `--use-wing-loss`: Usar Wing Loss ao invés de SmoothL1

**Parâmetros de Treinamento:**
- `--batch-size`: Tamanho do batch (padrão: 256)
- `--epochs`: Número de épocas (padrão: 30)
- `--lr`: Learning rate inicial (padrão: 0.1)
- `--momentum`: Momentum do SGD (padrão: 0.9)
- `--weight-decay`: Weight decay (padrão: 5e-4)

**Learning Rate Scheduler:**
- `--milestones`: Épocas para reduzir LR (padrão: [10, 20, 25])
- `--gamma`: Fator de decaimento do LR (padrão: 0.1)

**Validação:**
- `--lfw-root`: Diretório do dataset LFW (padrão: data/val)
- `--eval-freq`: Frequência de avaliação em épocas (padrão: 1)

**Saída:**
- Checkpoint do último modelo: `mobilenetv3_vggface2_multitask_antispoofing_last.ckpt`
- Checkpoint do melhor modelo: `mobilenetv3_vggface2_multitask_antispoofing_best.ckpt`
- Logs de treinamento com loss, accuracy e LFW results

### 3. Inferência

O script de inferência suporta quatro modos de operação, todos com detecção automática de spoofing.

#### 3.1 Comparação de Duas Faces

Compara duas imagens e determina se são da mesma pessoa, com verificação de liveness.

```bash
python inference_multitask.py \
    --mode compare \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_antispoofing_best.ckpt \
    --img1 path/to/image1.jpg \
    --img2 path/to/image2.jpg \
    --similarity-threshold 0.35 \
    --spoof-threshold 0.5
```

**Saída:**
- Valor de similaridade coseno
- Decisão binária (mesma pessoa ou não)
- Spoof score para cada imagem (0-1)
- Alerta se fake detectado
- Comparação com thresholds

#### 3.2 Extração de Embedding

Extrai o embedding de uma única imagem com verificação de spoofing.

```bash
python inference_multitask.py \
    --mode extract \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_antispoofing_best.ckpt \
    --img1 path/to/image.jpg \
    --spoof-threshold 0.5
```

**Saída:**
- Vetor de embedding (512D)
- Norma L2 do embedding
- Spoof score (probabilidade de ser fake)
- Status da detecção facial

#### 3.3 Processamento em Lote

Extrai embeddings de múltiplas imagens em uma pasta com filtragem de spoofs.

```bash
python inference_multitask.py \
    --mode batch \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_antispoofing_best.ckpt \
    --folder path/to/images/ \
    --output embeddings.json \
    --spoof-threshold 0.5
```

**Saída:**
- Arquivo JSON com embeddings e spoof scores
- Relatório de sucesso/falha/spoofs detectados
- Estatísticas de processamento

#### 3.4 Teste de Anti-Spoofing

Testa apenas a detecção de liveness, sem extrair embedding:

```bash
python inference_multitask.py \
    --mode test-spoof \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_antispoofing_best.ckpt \
    --img1 path/to/image.jpg \
    --spoof-threshold 0.5
```

**Saída:**
- Spoof score (0-1)
- Classificação (REAL ou FAKE)
- Confiança da predição

**Parâmetros Gerais:**
- `--checkpoint`: Caminho para o checkpoint do modelo
- `--similarity-threshold`: Limiar de similaridade para comparação (padrão: 0.35)
- `--spoof-threshold`: Limiar para classificar como fake (padrão: 0.5)
- `--disable-spoofing-check`: Desativa rejeição automática, apenas reporta scores
- `--gpu`: ID da GPU ou -1 para CPU (padrão: 0)
- `--mtcnn-thresholds`: Thresholds do MTCNN [pnet, rnet, onet] (padrão: [0.6, 0.7, 0.7])

## Pipeline de Pré-processamento

### Etapas de Processamento

1. **Detecção Facial**: MTCNN detecta faces e landmarks
2. **Validação**: Verifica tamanho mínimo e qualidade dos landmarks
3. **Alinhamento**: Transformação afim para normalizar pose facial
4. **Normalização de Landmarks**: Coordenadas normalizadas para [0, 1]
5. **Salvamento**: Imagens alinhadas e landmarks em JSON

### Critérios de Validação

- Tamanho mínimo da face detectada
- Landmarks dentro da bounding box
- Ordem vertical correta (olhos > nariz > boca)
- Alinhamento horizontal dos olhos

### Template de Alinhamento

Posições padrão dos landmarks em face frontal normalizada (112x112):

```
Left Eye:     (38.29, 51.70)
Right Eye:    (73.53, 51.50)
Nose:         (56.03, 71.74)
Mouth Left:   (41.55, 92.37)
Mouth Right:  (70.73, 92.20)
```

## Formato do Checkpoint

Os checkpoints salvos contêm:

```python
{
    'epoch': int,                      # Época atual
    'model': OrderedDict,              # Estado do modelo (inclui cabeça anti-spoofing)
    'classification_head': OrderedDict,# Estado do MCP head
    'optimizer': dict,                 # Estado do optimizer
    'lr_scheduler': dict,              # Estado do scheduler
    'best_lfw_accuracy': float,        # Melhor accuracy no LFW
    'num_classes': int,                # Número de identidades
    'args': Namespace                  # Argumentos de treinamento
}
```

## Formato do Landmarks JSON

Estrutura do arquivo de landmarks:

```json
{
    "identity_name/image_name.jpg": [
        x1, y1,  // Left Eye
        x2, y2,  // Right Eye
        x3, y3,  // Nose
        x4, y4,  // Mouth Left
        x5, y5   // Mouth Right
    ]
}
```

Todos os valores são normalizados para o intervalo [0, 1].

## Dataset Esperado

### VGGFace2

Estrutura do dataset:

```
data/
├── raw/
│   └── vggface2_112x112/
│       ├── n000001/
│       │   ├── 0001_01.jpg
│       │   ├── 0002_01.jpg
│       │   └── ...
│       ├── n000002/
│       └── ...
├── train/
│   ├── vggface2_aligned_112x112/
│   └── vggface2_landmarks.json
```

- Aproximadamente 8.631 identidades
- Imagens pré-redimensionadas para 112x112 (opcional)
- Múltiplas imagens por identidade

### LFW (Validação)

```
├── val/
│    └── lfw/
│       ├── Aaron_Eckhart/
│       │   └── Aaron_Eckhart0001.jpg
│       ├── Aaron_Guiel/
│       └── ...
    └──pairs.txt
```

Dataset para validação de verificação facial:
- 6.000 pares de faces
- Split padrão para benchmark

### CASIA-FASD (Anti-Spoofing)

```
├── casia-fasd/
│   ├── train/
│   │   ├── live/     # Imagens reais
│   │   └── spoof/    # Ataques (foto, vídeo, máscara)
│   └── test/
│       ├── live/
│       └── spoof/
```

Dataset para treinamento de detecção de liveness:
- Múltiplos tipos de ataque
- Vídeos e imagens estáticas

## Métricas de Avaliação

### Treinamento

- **Loss Total**: Soma ponderada de classificação, landmarks e spoofing
- **Loss de Classificação**: Cross-entropy via Margin Cosine Product
- **Loss de Landmarks**: SmoothL1 ou Wing Loss
- **Loss de Spoofing**: Binary Cross-Entropy para detecção de liveness
- **Accuracy**: Precisão de classificação de identidades
- **Spoof Accuracy**: Precisão na detecção de fakes

### Validação

- **LFW Accuracy**: Taxa de verificação correta no dataset LFW
- **Curvas ROC**: True Positive Rate vs False Positive Rate
- **TAR @ FAR**: True Accept Rate em diferentes False Accept Rates
