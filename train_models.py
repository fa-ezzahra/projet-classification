#!/usr/bin/env python3
"""
Script d'entraînement complet pour les modèles CV et NLP
Version améliorée avec extraction OCR automatique
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "src"))

from src.computer_vision.cv_model import HybridCVModel, CVModelTrainer
from src.nlp.nlp_model import CamembertClassifier, NLPModelTrainer, create_text_dataset
from src.nlp.ocr_extractor import OCRExtractor
from transformers import CamembertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_cv_dataset(data_dir: str, gabarits_detector):
    """
    Prépare le dataset pour l'entraînement CV
    
    Structure attendue:
    data_dir/
        piece_identite/
            image1.png
            image2.png
        releve_bancaire/
            image1.png
        ...
    
    Args:
        data_dir: Dossier contenant les images
        gabarits_detector: Détecteur de gabarits
        
    Returns:
        Tuple (images, gabarit_features, labels)
    """
    from PIL import Image
    import cv2
    
    data_path = Path(data_dir)
    
    classes = [
        "piece_identite",
        "releve_notes",
        "facture"
    ]
    
    all_images = []
    all_features = []
    all_labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = data_path / class_name
        
        if not class_path.exists():
            logger.warning(f"Dossier {class_path} non trouvé, ignoré")
            continue
        
        image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg"))
        
        logger.info(f"Classe {class_name}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"Chargement {class_name}"):
            # Charger l'image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Extraire les features de gabarits
            features = gabarits_detector.extract_all_features(img, "")
            
            # Convertir features en array
            feature_order = [
                'aspect_ratio', 'has_photo', 'photo_confidence',
                'has_table', 'table_confidence', 'text_density',
                'numeric_density', 'has_signature', 'signature_confidence'
            ]
            feature_array = np.array([features.get(f, 0.0) for f in feature_order], dtype=np.float32)
            
            # Normaliser aspect_ratio
            feature_array[0] = min(max(feature_array[0], 0.5), 2.0) / 2.0
            
            all_images.append(img)
            all_features.append(feature_array)
            all_labels.append(class_idx)
    
    logger.info(f"Total: {len(all_images)} images chargées")
    
    return all_images, all_features, all_labels


def prepare_nlp_dataset_from_images(data_dir: str):
    """
    Prépare le dataset NLP en extrayant le texte des images via OCR
    (NOUVEAU: remplace l'ancienne fonction qui attendait des .txt)
    
    Structure attendue:
    data_dir/
        piece_identite/
            image1.png
            image2.png
        releve_notes/
            image1.png
        ...
    
    Args:
        data_dir: Dossier contenant les images
        
    Returns:
        Tuple (texts, labels)
    """
    import cv2
    
    data_path = Path(data_dir)
    ocr_extractor = OCRExtractor()
    
    classes = [
        "piece_identite",
        "releve_notes",
        "facture"
    ]
    
    all_texts = []
    all_labels = []
    
    logger.info("Extraction du texte via OCR...")
    
    for class_idx, class_name in enumerate(classes):
        class_path = data_path / class_name
        
        if not class_path.exists():
            logger.warning(f"Dossier {class_path} non trouvé, ignoré")
            continue
        
        image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg"))
        
        logger.info(f"Classe {class_name}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"OCR {class_name}"):
            try:
                # Charger l'image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Extraire le texte
                ocr_result = ocr_extractor.process_document(img)
                text = ocr_result['corrected_text']
                
                if len(text.strip()) < 10:  # Ignorer les textes trop courts
                    logger.warning(f"Texte trop court pour {img_file.name}, ignoré")
                    continue
                
                all_texts.append(text)
                all_labels.append(class_idx)
                
            except Exception as e:
                logger.error(f"Erreur OCR pour {img_file}: {e}")
                continue
    
    logger.info(f"Total: {len(all_texts)} textes extraits")
    
    return all_texts, all_labels


def prepare_nlp_dataset_from_txt(data_dir: str):
    """
    Prépare le dataset NLP depuis des fichiers texte
    (Version originale, gardée pour compatibilité)
    
    Structure attendue:
    data_dir/
        piece_identite/
            text1.txt
            text2.txt
        releve_notes/
            text1.txt
        ...
    
    Args:
        data_dir: Dossier contenant les fichiers texte
        
    Returns:
        Tuple (texts, labels)
    """
    data_path = Path(data_dir)
    
    classes = [
        "piece_identite",
        "releve_notes",
        "facture"
    
    ]
    
    all_texts = []
    all_labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = data_path / class_name
        
        if not class_path.exists():
            logger.warning(f"Dossier {class_path} non trouvé, ignoré")
            continue
        
        text_files = list(class_path.glob("*.txt"))
        
        logger.info(f"Classe {class_name}: {len(text_files)} fichiers")
        
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if len(text) < 10:
                continue
            
            all_texts.append(text)
            all_labels.append(class_idx)
    
    logger.info(f"Total: {len(all_texts)} textes chargés")
    
    return all_texts, all_labels


def train_cv_model(data_dir: str, epochs: int = 50, batch_size: int = 4):
    """
    Entraîne le modèle Computer Vision
    
    Args:
        data_dir: Dossier des données d'entraînement
        epochs: Nombre d'époques
        batch_size: Taille du batch
    """
    logger.info("="*60)
    logger.info("ENTRAÎNEMENT MODÈLE COMPUTER VISION")
    logger.info("="*60)
    
    # Importer les modules nécessaires
    from src.gabarits.detector import GabaritsDetector
    from src.preprocessing.pdf_processor import PDFPreprocessor
    import torchvision.transforms as transforms
    
    # Initialiser le détecteur de gabarits
    gabarits_detector = GabaritsDetector()
    
    # Préparer le dataset
    logger.info("Chargement des données...")
    images, features, labels = prepare_cv_dataset(data_dir, gabarits_detector)
    
    if len(images) == 0:
        logger.error("Aucune image trouvée ! Vérifiez le dossier de données.")
        return
    
    # Transformer les images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convertir en tensors
    logger.info("Prétraitement des images...")
    image_tensors = torch.stack([transform(img) for img in tqdm(images, desc="Transformation")])
    feature_tensors = torch.FloatTensor(np.array(features))
    label_tensors = torch.LongTensor(labels)
    
    # Créer le dataset
    dataset = TensorDataset(image_tensors, feature_tensors, label_tensors)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {train_size} samples, Val: {val_size} samples")
    
    # Créer et entraîner le modèle
    model = HybridCVModel(num_classes=5, gabarit_features_dim=9)
    trainer = CVModelTrainer(model)
    
    logger.info("Début de l'entraînement...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path="models/cv/resnet50_hybrid.pth"  # Nom cohérent avec le système
    )
    
    logger.info("✓ Entraînement CV terminé !")


def train_nlp_model(data_dir: str, epochs: int = 30, batch_size: int = 4, use_ocr: bool = True):
    """
    Entraîne le modèle NLP
    
    Args:
        data_dir: Dossier des données d'entraînement
        epochs: Nombre d'époques
        batch_size: Taille du batch
        use_ocr: Si True, extrait le texte des images. Si False, charge des .txt
    """
    logger.info("="*60)
    logger.info("ENTRAÎNEMENT MODÈLE NLP")
    logger.info("="*60)
    
    # Préparer le dataset
    logger.info("Chargement des données...")
    
    if use_ocr:
        texts, labels = prepare_nlp_dataset_from_images(data_dir)
    else:
        texts, labels = prepare_nlp_dataset_from_txt(data_dir)
    
    if len(texts) == 0:
        logger.error("Aucun texte trouvé ! Vérifiez le dossier de données.")
        return
    
    # Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    # Créer le dataset
    logger.info("Tokenisation des textes...")
    dataset = create_text_dataset(texts, labels, tokenizer, max_length=512)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Train: {train_size} samples, Val: {val_size} samples")
    
    # Créer et entraîner le modèle
    model = CamembertClassifier(num_classes=5)
    trainer = NLPModelTrainer(model)
    
    logger.info("Début de l'entraînement...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path="models/nlp/camembert_classifier.pth"  # Nom cohérent avec le système
    )
    
    logger.info("✓ Entraînement NLP terminé !")


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(
        description='Entraînement des modèles CV et NLP'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['cv', 'nlp', 'both'],
        default='both',
        help='Modèle à entraîner (cv, nlp, ou both)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/augmented/train',
        help='Dossier contenant les données d\'entraînement'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Nombre d\'époques (défaut: 50 pour CV, 30 pour NLP)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Taille du batch (défaut: 4 pour CV, 4 pour NLP)'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Pour NLP: charger des .txt au lieu d\'extraire depuis les images'
    )
    
    args = parser.parse_args()
    
    # Vérifier que le dossier existe
    if not Path(args.data_dir).exists():
        logger.error(f"Dossier {args.data_dir} non trouvé !")
        logger.info("\nExécutez d'abord: python prepare_dataset.py")
        return 1
    
    # Entraîner les modèles
    if args.model in ['cv', 'both']:
        train_cv_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    if args.model in ['nlp', 'both']:
        train_nlp_model(
            data_dir=args.data_dir,
            epochs=args.epochs if args.model == 'nlp' else 30,
            batch_size=args.batch_size if args.model == 'nlp' else 8,
            use_ocr=not args.no_ocr
        )
    
    logger.info("\n" + "="*60)
    logger.info("✅ ENTRAÎNEMENT TERMINÉ")
    logger.info("="*60)
    logger.info("\nModèles sauvegardés:")
    if args.model in ['cv', 'both']:
        logger.info("  ✓ models/cv/resnet50_hybrid.pth")
    if args.model in ['nlp', 'both']:
        logger.info("  ✓ models/nlp/camembert_classifier.pth")
    
    logger.info("\nTestez maintenant avec:")
    logger.info("  python main.py --input data/raw --output outputs --verbose")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())