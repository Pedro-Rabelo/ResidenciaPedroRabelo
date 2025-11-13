import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from pathlib import Path

from models.mobilenetv3_multitask import mobilenetv3_large_multitask
from facenet_pytorch import MTCNN


def load_multitask_model(checkpoint_path, device):
    """
    Carrega modelo VGGFace2 treinado com multi-task learning + anti-spoofing
    
    Args:
        checkpoint_path: caminho para o checkpoint (.ckpt)
        device: torch device (cpu ou cuda)
    
    Returns:
        model: modelo carregado em modo de avaliação
    """
    print(f"📂 Carregando modelo de: {checkpoint_path}")
    
    # Cria modelo
    model = mobilenetv3_large_multitask(embedding_dim=512).to(device)
    
    # Carrega checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Carrega pesos
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Informações do checkpoint
    if 'epoch' in checkpoint:
        print(f"✓ Modelo do epoch {checkpoint['epoch']}")
    if 'best_lfw_accuracy' in checkpoint:
        print(f"✓ LFW accuracy: {checkpoint['best_lfw_accuracy']:.4f}")
    if 'num_classes' in checkpoint:
        print(f"✓ Treinado com {checkpoint['num_classes']:,} classes")
    
    print()
    
    return model


def align_face(img, landmarks):
    """
    Alinha face usando os 5 landmarks do MTCNN
    
    Args:
        img: imagem numpy array (RGB)
        landmarks: array [5, 2] com landmarks do MTCNN
    
    Returns:
        aligned: imagem alinhada 112x112
    """
    # Template padrão para face frontal (mesmo do treinamento)
    template = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041]   # right mouth
    ], dtype=np.float32)
    
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Calcula transformação afim
    tform = cv2.estimateAffinePartial2D(landmarks, template)[0]
    
    # Aplica transformação
    aligned = cv2.warpAffine(img, tform, (112, 112))
    
    return aligned


def extract_embedding(model, img_path, detector, device, 
                     spoof_threshold=0.5, check_spoofing=True, verbose=False):
    """
    Extrai embedding de uma imagem com detecção de spoofing
    
    Args:
        model: modelo carregado
        img_path: caminho para a imagem
        detector: detector de faces (MTCNN)
        device: torch device
        spoof_threshold: threshold para detecção de spoofing (default: 0.5)
        check_spoofing: se True, verifica se é spoof (default: True)
        verbose: se True, imprime informações
    
    Returns:
        embedding: vetor de features (512D) ou None se spoof detectado
        spoofing_score: probabilidade de ser spoof (0-1)
        status: mensagem de status
    """
    if verbose:
        print(f"  Processando: {img_path}")
    
    # Carrega imagem
    if not os.path.exists(img_path):
        return None, None, f"File not found: {img_path}"
    
    img = cv2.imread(img_path)
    if img is None:
        return None, None, "Failed to load image"
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # ========== MTCNN: Detecta faces e landmarks ==========
    boxes, probs, landmarks = detector.detect(img_pil, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        return None, None, "No face detected"
    
    # Usa a maior face se houver múltiplas
    if len(boxes) > 1:
        # Calcula área de cada bbox
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_idx = np.argmax(areas)
        box = boxes[largest_idx]
        landmark = landmarks[largest_idx]
        prob = probs[largest_idx]
        
        if verbose:
            print(f"  ⚠️  {len(boxes)} faces detectadas, usando a maior (conf: {prob:.3f})")
    else:
        box = boxes[0]
        landmark = landmarks[0]
        prob = probs[0]
        
        if verbose:
            print(f"  ✓ Face detectada (conf: {prob:.3f})")
    
    # MTCNN landmarks já vêm no formato correto: [[x1,y1], [x2,y2], ...]
    # Ordem: left_eye, right_eye, nose, mouth_left, mouth_right
    landmarks_array = np.array(landmark, dtype=np.float32)
    
    # Alinha face
    aligned = align_face(img_rgb, landmarks_array)
    
    # Transforma para tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(Image.fromarray(aligned)).unsqueeze(0).to(device)
    
    # ========== Extrai embedding + score anti-spoofing ==========
    with torch.no_grad():
        embedding, spoofing_prob = model.extract_features(
            img_tensor, 
            return_spoofing_score=True
        )
    
    spoofing_score = spoofing_prob.item()
    
    if verbose:
        print(f"  ✓ Embedding extraído: {embedding.shape}")
        print(f"  🔍 Spoofing score: {spoofing_score:.4f} {'(FAKE!)' if spoofing_score > spoof_threshold else '(real)'}")
    
    # ========== Verifica se é spoof ==========
    if check_spoofing and spoofing_score > spoof_threshold:
        return None, spoofing_score, f"Spoofing detected (score: {spoofing_score:.4f})"
    
    return embedding.cpu().numpy(), spoofing_score, "Success"


def compute_similarity(embedding1, embedding2):
    """
    Calcula similaridade coseno entre dois embeddings
    
    Args:
        embedding1: primeiro embedding
        embedding2: segundo embedding
    
    Returns:
        similarity: similaridade no intervalo [-1, 1]
    """
    dot_product = np.dot(embedding1, embedding2.T)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2 + 1e-8)
    return similarity[0][0]


# ========== NOVO: MODO SINGLE IMAGE ==========

def test_single_image(model, detector, device, img_path, spoof_threshold=0.5):
    """
    Testa uma única imagem para verificar se é real ou fake
    
    Args:
        model: modelo carregado
        detector: MTCNN detector
        device: torch device
        img_path: caminho da imagem
        spoof_threshold: threshold de spoofing (default: 0.5)
    
    Returns:
        result: dict com informações
    """
    print("="*60)
    print("SINGLE IMAGE ANTI-SPOOFING TEST")
    print("="*60)
    print(f"Image: {img_path}")
    print(f"Threshold: {spoof_threshold}\n")
    
    # Extrai embedding e spoof score
    _, spoof_score, status = extract_embedding(
        model, img_path, detector, device,
        spoof_threshold=spoof_threshold,
        check_spoofing=False,  # Não rejeita, apenas retorna score
        verbose=True
    )
    
    if spoof_score is None:
        print(f"\n❌ Error: {status}")
        return None
    
    is_spoof = spoof_score > spoof_threshold
    confidence = abs(spoof_score - spoof_threshold)
    
    # Resultado
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"Spoof Score:    {spoof_score:.4f}")
    print(f"Threshold:      {spoof_threshold:.4f}")
    print(f"Classification: {'🚨 FAKE' if is_spoof else '✅ REAL'}")
    print(f"Confidence:     {confidence:.4f}")
    print(f"{'='*60}\n")
    
    result = {
        'image': img_path,
        'spoof_score': spoof_score,
        'threshold': spoof_threshold,
        'is_fake': is_spoof,
        'confidence': confidence,
        'status': 'FAKE' if is_spoof else 'REAL'
    }
    
    return result


# ========== NOVO: MODO PAIR (COMPARAÇÃO COM SPOOFING) ==========

def compare_pair_with_spoofing(model, detector, device, img1_path, img2_path,
                               similarity_threshold=0.35, spoof_threshold=0.5):
    """
    Compara par de imagens retornando:
    1. Similaridade (mesma pessoa ou não)
    2. Probabilidade de cada imagem ser real/fake
    
    Args:
        model: modelo carregado
        detector: MTCNN detector
        device: torch device
        img1_path: caminho primeira imagem
        img2_path: caminho segunda imagem
        similarity_threshold: threshold de similaridade (default: 0.35)
        spoof_threshold: threshold de spoofing (default: 0.5)
    
    Returns:
        result: dict com todas as informações
    """
    print("="*60)
    print("PAIR COMPARISON WITH ANTI-SPOOFING")
    print("="*60)
    
    # ========== Processa Imagem 1 ==========
    print(f"\n📸 Image 1: {img1_path}")
    emb1, spoof1, status1 = extract_embedding(
        model, img1_path, detector, device,
        spoof_threshold=spoof_threshold,
        check_spoofing=False,  # Não rejeita, extrai tudo
        verbose=True
    )
    
    if emb1 is None:
        print(f"❌ Error: {status1}")
        return None
    
    # ========== Processa Imagem 2 ==========
    print(f"\n📸 Image 2: {img2_path}")
    emb2, spoof2, status2 = extract_embedding(
        model, img2_path, detector, device,
        spoof_threshold=spoof_threshold,
        check_spoofing=False,
        verbose=True
    )
    
    if emb2 is None:
        print(f"❌ Error: {status2}")
        return None
    
    # ========== Calcula Similaridade ==========
    similarity = compute_similarity(emb1, emb2)
    is_same_person = similarity > similarity_threshold
    
    # ========== Análise de Spoofing ==========
    img1_is_fake = spoof1 > spoof_threshold
    img2_is_fake = spoof2 > spoof_threshold
    both_real = not img1_is_fake and not img2_is_fake
    
    # ========== Resultado Completo ==========
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    
    print(f"\n--- Similarity Analysis ---")
    print(f"Similarity Score:   {similarity:.4f}")
    print(f"Threshold:          {similarity_threshold:.4f}")
    print(f"Same Person:        {'✅ YES' if is_same_person else '❌ NO'}")
    
    print(f"\n--- Anti-Spoofing Analysis ---")
    print(f"Image 1:")
    print(f"  Spoof Score:      {spoof1:.4f}")
    print(f"  Classification:   {'🚨 FAKE' if img1_is_fake else '✅ REAL'}")
    
    print(f"Image 2:")
    print(f"  Spoof Score:      {spoof2:.4f}")
    print(f"  Classification:   {'🚨 FAKE' if img2_is_fake else '✅ REAL'}")
    
    print(f"\nSpoof Threshold:    {spoof_threshold:.4f}")
    
    print(f"\n--- Final Decision ---")
    if both_real:
        if is_same_person:
            decision = "✅ VERIFIED: Same person, both images are real"
        else:
            decision = "❌ REJECTED: Different people (both real)"
    else:
        decision = "🚨 REJECTED: One or both images are FAKE"
    
    print(f"{decision}")
    print(f"{'='*60}\n")
    
    result = {
        'similarity': {
            'score': similarity,
            'threshold': similarity_threshold,
            'same_person': is_same_person
        },
        'image1': {
            'path': img1_path,
            'spoof_score': spoof1,
            'is_fake': img1_is_fake,
            'status': 'FAKE' if img1_is_fake else 'REAL'
        },
        'image2': {
            'path': img2_path,
            'spoof_score': spoof2,
            'is_fake': img2_is_fake,
            'status': 'FAKE' if img2_is_fake else 'REAL'
        },
        'spoof_threshold': spoof_threshold,
        'final_decision': decision
    }
    
    return result


# ========== MODO COMPARE (ANTIGO, MANTIDO PARA COMPATIBILIDADE) ==========

def compare_two_faces(model, detector, device, img1_path, img2_path, 
                     similarity_threshold=0.35, spoof_threshold=0.5,
                     check_spoofing=True):
    """
    Compara duas faces e determina se são da mesma pessoa
    Agora com detecção de spoofing integrada!
    
    Args:
        model: modelo carregado
        detector: detector de faces
        device: torch device
        img1_path: caminho para primeira imagem
        img2_path: caminho para segunda imagem
        similarity_threshold: limiar de similaridade (default: 0.35)
        spoof_threshold: limiar de spoofing (default: 0.5)
        check_spoofing: se True, rejeita imagens com spoof (default: True)
    
    Returns:
        similarity: valor de similaridade (ou None se spoof detectado)
        is_same: True se mesma pessoa, False caso contrário
        spoof_info: dict com informações de spoofing
    """
    print("="*60)
    print("COMPARAÇÃO DE FACES COM ANTI-SPOOFING")
    print("="*60)
    
    spoof_info = {
        'img1_spoof_score': None,
        'img2_spoof_score': None,
        'img1_is_spoof': False,
        'img2_is_spoof': False
    }
    
    # ========== Extrai embedding 1 ==========
    print(f"\n📸 Imagem 1: {img1_path}")
    emb1, spoof_score1, status1 = extract_embedding(
        model, img1_path, detector, device,
        spoof_threshold=spoof_threshold,
        check_spoofing=check_spoofing,
        verbose=True
    )
    
    spoof_info['img1_spoof_score'] = spoof_score1
    spoof_info['img1_is_spoof'] = spoof_score1 is not None and spoof_score1 > spoof_threshold
    
    if emb1 is None:
        print(f"❌ Erro: {status1}")
        return None, None, spoof_info
    
    # ========== Extrai embedding 2 ==========
    print(f"\n📸 Imagem 2: {img2_path}")
    emb2, spoof_score2, status2 = extract_embedding(
        model, img2_path, detector, device,
        spoof_threshold=spoof_threshold,
        check_spoofing=check_spoofing,
        verbose=True
    )
    
    spoof_info['img2_spoof_score'] = spoof_score2
    spoof_info['img2_is_spoof'] = spoof_score2 is not None and spoof_score2 > spoof_threshold
    
    if emb2 is None:
        print(f"❌ Erro: {status2}")
        return None, None, spoof_info
    
    # ========== Calcula similaridade ==========
    similarity = compute_similarity(emb1, emb2)
    is_same = similarity > similarity_threshold
    
    # ========== Resultado ==========
    print(f"\n{'='*60}")
    print(f"RESULTADO")
    print(f"{'='*60}")
    print(f"Similaridade:       {similarity:.4f}")
    print(f"Threshold:          {similarity_threshold:.4f}")
    print(f"Mesma pessoa:       {'✅ SIM' if is_same else '❌ NÃO'}")
    print(f"\n--- Anti-Spoofing ---")
    print(f"Img1 spoof score:   {spoof_score1:.4f} {'🚨 FAKE' if spoof_info['img1_is_spoof'] else '✅ real'}")
    print(f"Img2 spoof score:   {spoof_score2:.4f} {'🚨 FAKE' if spoof_info['img2_is_spoof'] else '✅ real'}")
    print(f"Spoof threshold:    {spoof_threshold:.4f}")
    print(f"{'='*60}\n")
    
    return similarity, is_same, spoof_info


def batch_extract_embeddings(model, detector, device, image_folder, 
                             output_file=None, spoof_threshold=0.5,
                             check_spoofing=True):
    """
    Extrai embeddings de múltiplas imagens em uma pasta
    Agora com filtragem de spoofs!
    
    Args:
        model: modelo carregado
        detector: detector de faces (MTCNN)
        device: torch device
        image_folder: pasta com imagens
        output_file: arquivo para salvar embeddings (opcional)
        spoof_threshold: threshold de spoofing (default: 0.5)
        check_spoofing: se True, filtra imagens fake (default: True)
    
    Returns:
        embeddings_dict: dicionário {filename: {"embedding": [...], "spoof_score": 0.xx}}
    """
    print("="*60)
    print(f"EXTRAÇÃO EM LOTE DE EMBEDDINGS (COM ANTI-SPOOFING)")
    print("="*60)
    print(f"Pasta: {image_folder}")
    print(f"Anti-spoofing: {'ATIVO' if check_spoofing else 'DESATIVADO'}")
    print(f"Spoof threshold: {spoof_threshold}\n")
    
    image_folder = Path(image_folder)
    
    # Lista imagens
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
    
    print(f"📊 Encontradas {len(image_files)} imagens\n")
    
    embeddings_dict = {}
    success_count = 0
    spoof_detected_count = 0
    
    for img_path in image_files:
        embedding, spoof_score, status = extract_embedding(
            model, str(img_path), detector, device,
            spoof_threshold=spoof_threshold,
            check_spoofing=check_spoofing,
            verbose=False
        )
        
        if embedding is not None:
            embeddings_dict[img_path.name] = {
                "embedding": embedding.tolist(),
                "spoof_score": spoof_score
            }
            success_count += 1
            print(f"✓ {img_path.name} (spoof: {spoof_score:.3f})")
        else:
            if "Spoofing detected" in status:
                spoof_detected_count += 1
                print(f"🚨 {img_path.name} - FAKE DETECTED (spoof: {spoof_score:.3f})")
            else:
                print(f"✗ {img_path.name} - {status}")
    
    print(f"\n{'='*60}")
    print(f"Sucesso: {success_count}/{len(image_files)}")
    print(f"Spoofs detectados: {spoof_detected_count}")
    print(f"{'='*60}\n")
    
    # Salva em arquivo se especificado
    if output_file and embeddings_dict:
        import json
        with open(output_file, 'w') as f:
            json.dump(embeddings_dict, f, indent=2)
        print(f"✅ Embeddings salvos em: {output_file}\n")
    
    return embeddings_dict


def test_antispoofing_only(model, detector, device, img_path, spoof_threshold=0.5):
    """
    Testa apenas a detecção de spoofing (sem extrair embedding)
    
    Args:
        model: modelo carregado
        detector: detector de faces (MTCNN)
        device: torch device
        img_path: caminho da imagem
        spoof_threshold: threshold (default: 0.5)
    
    Returns:
        spoof_score: probabilidade de ser fake (0-1)
        is_spoof: True se detectado como fake
    """
    print("="*60)
    print("TESTE DE ANTI-SPOOFING")
    print("="*60)
    print(f"Imagem: {img_path}")
    print(f"Threshold: {spoof_threshold}\n")
    
    _, spoof_score, status = extract_embedding(
        model, img_path, detector, device,
        spoof_threshold=spoof_threshold,
        check_spoofing=False,  # Não rejeita, apenas retorna score
        verbose=True
    )
    
    if spoof_score is None:
        print(f"\n❌ Erro: {status}")
        return None, None
    
    is_spoof = spoof_score > spoof_threshold
    
    print(f"\n{'='*60}")
    print(f"RESULTADO")
    print(f"{'='*60}")
    print(f"Spoof score:  {spoof_score:.4f}")
    print(f"Threshold:    {spoof_threshold:.4f}")
    print(f"Classificação: {'🚨 FAKE DETECTED' if is_spoof else '✅ REAL'}")
    print(f"Confiança:     {abs(spoof_score - spoof_threshold):.4f}")
    print(f"{'='*60}\n")
    
    return spoof_score, is_spoof


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Inferência com modelo VGGFace2 + Anti-Spoofing (usando MTCNN)"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['compare', 'extract', 'batch', 'test-spoof', 'single', 'pair'],
        default='pair',
        help='Modo: '
             'single (test single image real/fake), '
             'pair (compare 2 images + spoofing), '
             'compare (deprecated, use pair), '
             'extract (extract embedding), '
             'batch (folder), '
             'test-spoof (spoofing only)'
    )
    
    # Paths
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='weights/vggface2/mobilenetv3_vggface2_multitask_antispoofing_best.ckpt',
        help='Caminho para o checkpoint do modelo'
    )
    parser.add_argument(
        '--img1',
        type=str,
        help='Caminho para primeira imagem (mode=compare/extract/test-spoof/single/pair)'
    )
    parser.add_argument(
        '--img2',
        type=str,
        help='Caminho para segunda imagem (mode=compare/pair)'
    )
    parser.add_argument(
        '--folder',
        type=str,
        help='Pasta com imagens (mode=batch)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Arquivo de saída para embeddings (mode=batch)'
    )
    
    # Parâmetros Anti-Spoofing
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.35,
        help='Threshold de similaridade (default: 0.35)'
    )
    parser.add_argument(
        '--spoof-threshold',
        type=float,
        default=0.5,
        help='Threshold de spoofing (default: 0.5, range: 0-1)'
    )
    parser.add_argument(
        '--disable-spoofing-check',
        action='store_true',
        help='Desativa verificação de spoofing (apenas extrai scores)'
    )
    
    # GPU
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID (-1 para CPU, default: 0)'
    )
    
    # MTCNN params
    parser.add_argument(
        '--mtcnn-thresholds',
        type=float,
        nargs=3,
        default=[0.6, 0.7, 0.7],
        help='MTCNN thresholds [pnet, rnet, onet] (default: [0.6, 0.7, 0.7])'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    print(f"\n🖥️  Device: {device}\n")
    
    # Carrega modelo
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint não encontrado: {args.checkpoint}")
        return
    
    model = load_multitask_model(args.checkpoint, device)
    
    # ========== Inicializa MTCNN ==========
    print("📷 Inicializando MTCNN...")
    detector = MTCNN(
        image_size=112,
        margin=0,
        min_face_size=20,
        thresholds=args.mtcnn_thresholds,
        factor=0.709,
        post_process=False,
        device=device,
        keep_all=True  # Detecta múltiplas faces
    )
    print("✓ MTCNN pronto\n")
    
    check_spoofing = not args.disable_spoofing_check
    
    # ========== Executa modo selecionado ==========
    
    # NOVO: Modo Single Image
    if args.mode == 'single':
        if not args.img1:
            print("❌ Mode 'single' requires --img1")
            return
        
        test_single_image(
            model, detector, device,
            args.img1,
            spoof_threshold=args.spoof_threshold
        )
    
    # NOVO: Modo Pair (substitui compare)
    elif args.mode == 'pair':
        if not args.img1 or not args.img2:
            print("❌ Mode 'pair' requires --img1 and --img2")
            return
        
        compare_pair_with_spoofing(
            model, detector, device,
            args.img1, args.img2,
            similarity_threshold=args.similarity_threshold,
            spoof_threshold=args.spoof_threshold
        )
    
    # Modo Compare (antigo, mantido para compatibilidade)
    elif args.mode == 'compare':
        if not args.img1 or not args.img2:
            print("❌ Mode 'compare' requer --img1 e --img2")
            return
        
        compare_two_faces(
            model, detector, device,
            args.img1, args.img2,
            similarity_threshold=args.similarity_threshold,
            spoof_threshold=args.spoof_threshold,
            check_spoofing=check_spoofing
        )
    
    elif args.mode == 'extract':
        if not args.img1:
            print("❌ Mode 'extract' requer --img1")
            return
        
        print(f"📸 Extraindo embedding de: {args.img1}\n")
        embedding, spoof_score, status = extract_embedding(
            model, args.img1, detector, device,
            spoof_threshold=args.spoof_threshold,
            check_spoofing=check_spoofing,
            verbose=True
        )
        
        if embedding is not None:
            print(f"\n✅ Embedding extraído com sucesso!")
            print(f"Shape: {embedding.shape}")
            print(f"Norma L2: {np.linalg.norm(embedding):.4f}")
            print(f"Spoof score: {spoof_score:.4f}")
        else:
            print(f"\n❌ Falha: {status}")
    
    elif args.mode == 'batch':
        if not args.folder:
            print("❌ Mode 'batch' requer --folder")
            return
        
        batch_extract_embeddings(
            model, detector, device,
            args.folder,
            output_file=args.output,
            spoof_threshold=args.spoof_threshold,
            check_spoofing=check_spoofing
        )
    
    elif args.mode == 'test-spoof':
        if not args.img1:
            print("❌ Mode 'test-spoof' requer --img1")
            return
        
        test_antispoofing_only(
            model, detector, device,
            args.img1,
            spoof_threshold=args.spoof_threshold
        )


if __name__ == "__main__":
    main()