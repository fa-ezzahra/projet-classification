#!/usr/bin/env python3
"""
Script d'initialisation offline pour télécharger tous les modèles nécessaires
À exécuter une seule fois avec connexion internet
"""
import os
import torch
import torchvision
from transformers import CamembertModel, CamembertTokenizer
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_cv_models(models_dir: Path):
    """
    Télécharge les modèles Computer Vision
    """
    logger.info("Téléchargement des modèles Computer Vision...")
    
    cv_dir = models_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # ResNet50
    logger.info("Téléchargement de ResNet50...")
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    torch.save(model.state_dict(), cv_dir / "resnet50.pth")
    logger.info("✓ ResNet50 téléchargé")
    
    # EfficientNet-B0
    logger.info("Téléchargement d'EfficientNet-B0...")
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), cv_dir / "efficientnet_b0.pth")
    logger.info("✓ EfficientNet-B0 téléchargé")


def download_nlp_models(models_dir: Path):
    """
    Télécharge les modèles NLP
    """
    logger.info("Téléchargement des modèles NLP...")
    
    nlp_dir = models_dir / "nlp"
    nlp_dir.mkdir(parents=True, exist_ok=True)
    
    # CamemBERT
    logger.info("Téléchargement de CamemBERT...")
    model_name = "camembert-base"
    model_path = nlp_dir / model_name
    model_path.mkdir(exist_ok=True)
    
    model = CamembertModel.from_pretrained(model_name)
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info("✓ CamemBERT téléchargé")


def verify_tesseract():
    """
    Vérifie que Tesseract OCR est installé
    """
    logger.info("Vérification de Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        logger.info(f"✓ Tesseract {version} détecté")
        
        # Vérifier la langue française
        try:
            pytesseract.get_languages()
            logger.info("✓ Langues Tesseract vérifiées")
        except Exception as e:
            logger.warning(f"Impossible de vérifier les langues Tesseract: {e}")
            logger.warning("Assurez-vous que le pack de langue française est installé")
        
        return True
    except Exception as e:
        logger.error(f"✗ Tesseract non trouvé: {e}")
        logger.error("Veuillez installer Tesseract OCR:")
        logger.error("  - Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-fra")
        logger.error("  - macOS: brew install tesseract tesseract-lang")
        logger.error("  - Windows: Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def verify_dependencies():
    """
    Vérifie toutes les dépendances
    """
    logger.info("Vérification des dépendances...")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'Transformers',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'yaml': 'PyYAML',
        'pdf2image': 'PDF2Image',
        'pytesseract': 'PyTesseract'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} manquant")
            missing.append(name)
    
    if missing:
        logger.error("\nDépendances manquantes:")
        for dep in missing:
            logger.error(f"  - {dep}")
        logger.error("\nInstallez-les avec: pip install -r requirements.txt")
        return False
    
    return True


def create_directory_structure():
    """
    Crée la structure de dossiers nécessaire
    """
    logger.info("Création de la structure de dossiers...")
    
    directories = [
        "models/cv",
        "models/nlp",
        "models/gabarits",
        "data/raw",
        "data/processed",
        "data/annotations",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ {directory}")


def benchmark_models():
    """
    Benchmarking basique des modèles
    """
    logger.info("\nBenchmarking des modèles...")
    
    import time
    import numpy as np
    
    # Test ResNet50
    logger.info("Test ResNet50...")
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Warm-up
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # en ms
    logger.info(f"  Temps moyen d'inférence: {avg_time:.2f} ms")
    
    # Taille du modèle
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Nombre de paramètres: {param_count:,}")


def main():
    """
    Script principal d'initialisation
    """
    logger.info("="*60)
    logger.info("INITIALISATION OFFLINE - CLASSIFICATION DE DOCUMENTS")
    logger.info("="*60)
    
    # 1. Vérifier les dépendances
    logger.info("\n[1/5] Vérification des dépendances")
    if not verify_dependencies():
        logger.error("Veuillez installer les dépendances manquantes")
        return 1
    
    # 2. Vérifier Tesseract
    logger.info("\n[2/5] Vérification de Tesseract OCR")
    if not verify_tesseract():
        logger.error("Veuillez installer Tesseract OCR")
        return 1
    
    # 3. Créer la structure
    logger.info("\n[3/5] Création de la structure de dossiers")
    create_directory_structure()
    
    # 4. Télécharger les modèles
    logger.info("\n[4/5] Téléchargement des modèles")
    models_dir = Path("models")
    
    try:
        download_cv_models(models_dir)
        download_nlp_models(models_dir)
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des modèles: {e}")
        logger.error("Assurez-vous d'avoir une connexion internet")
        return 1
    
    # 5. Benchmarking
    logger.info("\n[5/5] Benchmarking des modèles")
    try:
        benchmark_models()
    except Exception as e:
        logger.warning(f"Benchmarking échoué: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("INITIALISATION TERMINÉE AVEC SUCCÈS")
    logger.info("="*60)
    logger.info("\nVous pouvez maintenant utiliser le système offline avec:")
    logger.info("  python main.py --input data/raw --output outputs")
    logger.info("\nOu pour tester avec un seul fichier:")
    logger.info("  python main.py --file document.pdf --output outputs")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
