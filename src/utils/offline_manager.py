"""
Gestionnaire de modèles offline pour le chargement et la gestion des modèles pré-entraînés
"""
import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineModelManager:
    """
    Classe pour gérer le chargement et la mise en cache des modèles offline
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le gestionnaire de modèles
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                   self.config['performance']['use_gpu'] else 'cpu')
        
        # Cache pour les modèles chargés
        self.loaded_models: Dict[str, Any] = {}
        
        logger.info(f"OfflineModelManager initialisé avec device: {self.device}")
    
    def load_cv_model(self, model_name: str = "resnet50"):
        """
        Charge le modèle de Computer Vision hybride
        
        Args:
            model_name: Nom du modèle (resnet50, efficientnet, etc.)
            
        Returns:
            CVClassifier prêt à l'emploi
        """
        cache_key = f"cv_{model_name}"
        
        if cache_key in self.loaded_models:
            logger.info(f"Modèle CV {model_name} chargé depuis le cache")
            return self.loaded_models[cache_key]
        
        try:
            # Importer le CVClassifier
            from computer_vision.cv_model import CVClassifier
            
            # Chercher le modèle avec différents noms possibles
            possible_paths = [
                self.models_dir / "cv" / f"{model_name}_hybrid.pth",  # Nom avec _hybrid
                self.models_dir / "cv" / f"{model_name}.pth",         # Nom sans _hybrid
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    logger.info(f"Modèle trouvé à: {model_path}")
                    break
            
            if model_path:
                logger.info(f"Chargement du modèle CV entraîné depuis {model_path}")
                classifier = CVClassifier(
                    model_path=str(model_path),
                    device=str(self.device)
                )
            else:
                logger.warning(f"Aucun modèle entraîné trouvé dans {self.models_dir / 'cv'}")
                logger.warning("Chemins recherchés:")
                for path in possible_paths:
                    logger.warning(f"  - {path}")
                logger.warning("Utilisation du modèle pré-entraîné sans fine-tuning")
                classifier = CVClassifier(
                    model_path=None,
                    device=str(self.device)
                )
            
            # Sauvegarder dans le cache
            if self.config['performance']['cache_models']:
                self.loaded_models[cache_key] = classifier
            
            logger.info(f"Modèle CV {model_name} chargé avec succès")
            return classifier
            
        except ImportError as e:
            logger.error(f"Erreur d'import du module CV: {e}")
            logger.error("Vérifiez que computer_vision/cv_model.py existe")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle CV: {e}")
            raise
    
    def load_nlp_model(self, model_name: str = "camembert-base"):
        """
        Charge le modèle NLP (CamemBERT)
        
        Args:
            model_name: Nom du modèle NLP
            
        Returns:
            Tuple (model, tokenizer)
        """
        cache_key = f"nlp_{model_name}"
        
        if cache_key in self.loaded_models:
            logger.info(f"Modèle NLP {model_name} chargé depuis le cache")
            return self.loaded_models[cache_key]
        
        try:
            from transformers import CamembertModel, CamembertTokenizer
            
            model_path = str(self.models_dir / "nlp" / model_name)
            
            # Charger depuis local si disponible, sinon depuis HuggingFace
            try:
                model = CamembertModel.from_pretrained(model_path, local_files_only=True)
                tokenizer = CamembertTokenizer.from_pretrained(model_path, local_files_only=True)
                logger.info(f"Modèle NLP chargé depuis le stockage local")
            except:
                logger.warning(f"Modèle local non trouvé, téléchargement depuis HuggingFace")
                model = CamembertModel.from_pretrained(model_name)
                tokenizer = CamembertTokenizer.from_pretrained(model_name)
                
                # Sauvegarder localement pour utilisation future
                os.makedirs(model_path, exist_ok=True)
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
            
            model = model.to(self.device)
            model.eval()
            
            result = (model, tokenizer)
            
            # Sauvegarder dans le cache
            if self.config['performance']['cache_models']:
                self.loaded_models[cache_key] = result
            
            logger.info(f"Modèle NLP {model_name} chargé avec succès")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle NLP: {e}")
            raise
    
    def verify_dependencies(self) -> Dict[str, bool]:
        """
        Vérifie que toutes les dépendances nécessaires sont disponibles
        
        Returns:
            Dictionnaire avec le statut de chaque dépendance
        """
        status = {}
        
        # Vérifier PyTorch
        try:
            import torch
            status['torch'] = True
            status['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            status['torch'] = False
            status['cuda_available'] = False
        
        # Vérifier Transformers
        try:
            import transformers
            status['transformers'] = True
        except ImportError:
            status['transformers'] = False
        
        # Vérifier OpenCV
        try:
            import cv2
            status['opencv'] = True
        except ImportError:
            status['opencv'] = False
        
        # Vérifier Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            status['tesseract'] = True
        except:
            status['tesseract'] = False
        
        # Vérifier PDF2Image
        try:
            import pdf2image
            status['pdf2image'] = True
        except ImportError:
            status['pdf2image'] = False
        
        # Vérifier le module CV personnalisé
        try:
            from computer_vision.cv_model import CVClassifier
            status['cv_model'] = True
        except ImportError:
            status['cv_model'] = False
        
        logger.info(f"Vérification des dépendances: {status}")
        return status
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """
        Récupère les informations sur un modèle chargé
        
        Args:
            model_key: Clé du modèle dans le cache
            
        Returns:
            Dictionnaire avec les informations du modèle
        """
        if model_key not in self.loaded_models:
            return None
        
        model = self.loaded_models[model_key]
        
        info = {
            'key': model_key,
            'type': type(model).__name__,
            'device': str(self.device),
            'cached': True
        }
        
        if isinstance(model, torch.nn.Module):
            info['num_parameters'] = sum(p.numel() for p in model.parameters())
        
        return info
    
    def clear_cache(self):
        """
        Vide le cache des modèles
        """
        self.loaded_models.clear()
        logger.info("Cache des modèles vidé")


if __name__ == "__main__":
    # Test du gestionnaire
    manager = OfflineModelManager()
    
    # Vérifier les dépendances
    deps = manager.verify_dependencies()
    print("\nStatut des dépendances:")
    for dep, status in deps.items():
        print(f"  {dep}: {'✓' if status else '✗'}")
    
    # Test de chargement du modèle CV
    try:
        print("\nTest de chargement du modèle CV...")
        cv_model = manager.load_cv_model("resnet50")
        print("✓ Modèle CV chargé avec succès")
        print(f"  Type: {type(cv_model).__name__}")
    except Exception as e:
        print(f"✗ Erreur: {e}")