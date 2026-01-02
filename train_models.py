#!/usr/bin/env python3
"""
Script d'entraînement complet pour les modèles CV et NLP
Nécessite un dataset annoté
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import argparse

sys.path.append(str(Path(__file__).parent / "src"))

from src.computer_vision.cv_model import HybridCVModel, CVModelTrainer
from src.nlp.nlp_model import CamembertClassifier, NLPModelTrainer, create_text_dataset
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
        "releve_bancaire",
        "facture_electricite",
        "facture_eau",
        "document_employeur"
    ]
    
    all_images = []
    all_features = []
    all_labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = data_path / class_name
        
        if not class_path.exists():
            logger.warning(f"Dossier {class_path} non trouvé, ignoré")
            continue
        
        image_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
        
        logger.info(f"Classe {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
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
            
            all_images.append(img)
            all_features.append(feature_array)
            all_labels.append(class_idx)
    
    logger.info(f"Total: {len(all_images)} images chargées")
    
    return all_images, all_features, all_labels


def prepare_nlp_dataset(data_dir: str):
    """
    Prépare le dataset pour l'entraînement NLP
    
    Structure attendue:
    data_dir/
        piece_identite/
            text1.txt
            text2.txt
        releve_bancaire/
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
        "releve_bancaire",
        "facture_electricite",
        "facture_eau",
        "document_employeur"
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


def train_cv_model(data_dir: str, epochs: int = 50, batch_size: int = 16):
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convertir en tensors
    image_tensors = torch.stack([transform(img) for img in images])
    feature_tensors = torch.FloatTensor(np.array(features))
    label_tensors = torch.LongTensor(labels)
    
    # Créer le dataset
    dataset = TensorDataset(image_tensors, feature_tensors, label_tensors)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {train_size} samples, Val: {val_size} samples")
    
    # Créer et entraîner le modèle
    model = HybridCVModel(num_classes=5, gabarit_features_dim=9)
    trainer = CVModelTrainer(model)
    
    logger.info("Début de l'entraînement...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path="models/cv/hybrid_model_finetuned.pth"
    )
    
    logger.info("Entraînement CV terminé !")


def train_nlp_model(data_dir: str, epochs: int = 30, batch_size: int = 8):
    """
    Entraîne le modèle NLP
    
    Args:
        data_dir: Dossier des données d'entraînement
        epochs: Nombre d'époques
        batch_size: Taille du batch
    """
    logger.info("="*60)
    logger.info("ENTRAÎNEMENT MODÈLE NLP")
    logger.info("="*60)
    
    # Préparer le dataset
    logger.info("Chargement des données...")
    texts, labels = prepare_nlp_dataset(data_dir)
    
    if len(texts) == 0:
        logger.error("Aucun texte trouvé ! Vérifiez le dossier de données.")
        return
    
    # Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    # Créer le dataset
    dataset = create_text_dataset(texts, labels, tokenizer, max_length=512)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {train_size} samples, Val: {val_size} samples")
    
    # Créer et entraîner le modèle
    model = CamembertClassifier(num_classes=5)
    trainer = NLPModelTrainer(model)
    
    logger.info("Début de l'entraînement...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path="models/nlp/camembert_finetuned.pth"
    )
    
    logger.info("Entraînement NLP terminé !")


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
        required=True,
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
        default=16,
        help='Taille du batch (défaut: 16 pour CV, 8 pour NLP)'
    )
    
    args = parser.parse_args()
    
    # Vérifier que le dossier existe
    if not Path(args.data_dir).exists():
        logger.error(f"Dossier {args.data_dir} non trouvé !")
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
            batch_size=args.batch_size if args.model == 'nlp' else 8
        )
    
    logger.info("\n" + "="*60)
    logger.info("ENTRAÎNEMENT TERMINÉ")
    logger.info("="*60)
    logger.info("\nModèles sauvegardés:")
    if args.model in ['cv', 'both']:
        logger.info("  - models/cv/hybrid_model_finetuned.pth")
    if args.model in ['nlp', 'both']:
        logger.info("  - models/nlp/camembert_finetuned.pth")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
