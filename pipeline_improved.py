"""
Pipeline personnalisé utilisant multimodal_fusion_improved.py
Sans toucher à l'ancien multimodal_fusion.py
"""
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict
import logging
import yaml
from tqdm import tqdm

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing.pdf_processor import PDFPreprocessor
from src.gabarits.detector import GabaritsDetector
from src.nlp.ocr_extractor import OCRExtractor
from src.utils.offline_manager import OfflineModelManager

# IMPORTANT: Importer la version améliorée
from multimodal_fusion_improved import MultimodalFusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedDocumentClassificationPipeline:
    """
    Pipeline avec gestion des cas spéciaux (documents vides, illisibles, hors classes)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le pipeline amélioré
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        logger.info("Initialisation du pipeline AMÉLIORÉ...")
        
        # Charger la configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialiser les composants
        self.model_manager = OfflineModelManager(config_path)
        self.pdf_processor = PDFPreprocessor(dpi=self.config['ocr']['dpi'])
        self.gabarit_detector = GabaritsDetector()
        self.ocr_extractor = OCRExtractor(
            lang=self.config['ocr']['language'],
            psm=self.config['ocr']['psm'],
            oem=self.config['ocr']['oem']
        )
        
        # IMPORTANT: Utiliser la fusion AMÉLIORÉE
        self.fusion = MultimodalFusion(config_path)
        
        # Charger les modèles CV et NLP
        logger.info("Chargement des modèles...")
        try:
            self.cv_model = self.model_manager.load_cv_model(
                self.config['computer_vision']['backbone']
            )
            self.nlp_model, self.nlp_tokenizer = self.model_manager.load_nlp_model(
                self.config['nlp']['model_name']
            )
            logger.info("Modèles chargés avec succès")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des modèles: {e}")
            logger.warning("Le pipeline fonctionnera en mode dégradé")
            self.cv_model = None
            self.nlp_model = None
            self.nlp_tokenizer = None
        
        self.classes = self.config['classes']
        
        logger.info("Pipeline AMÉLIORÉ initialisé avec succès")
        logger.info("✨ Gestion des cas spéciaux activée: vides, illisibles, hors classes")
    
    def extract_cv_features(self, image: np.ndarray, gabarit_features: Dict) -> Dict[str, float]:
        """
        Extrait les features Computer Vision
        """
        try:
            if self.cv_model is not None and hasattr(self.cv_model, 'predict'):
                return self.cv_model.predict(image, gabarit_features)
            else:
                logger.warning("Modèle CV non chargé, utilisation des gabarits")
                gabarit_scores = self.gabarit_detector.classify_by_gabarit(image, "")
                return gabarit_scores
        except Exception as e:
            logger.error(f"Erreur extraction CV features: {e}")
            return {cls: 0.2 for cls in self.classes}
    
    def extract_nlp_features(self, text: str) -> Dict[str, float]:
        """
        Extrait les features NLP
        """
        try:
            if self.nlp_model is not None and self.nlp_tokenizer is not None:
                from src.nlp.nlp_model import NLPClassifier
                nlp_classifier = NLPClassifier()
                return nlp_classifier.predict(text)
            else:
                logger.warning("Modèle NLP non chargé, utilisation des mots-clés")
                keyword_scores = self.ocr_extractor.calculate_keyword_scores(text)
                return keyword_scores
        except Exception as e:
            logger.error(f"Erreur extraction NLP features: {e}")
            keyword_scores = self.ocr_extractor.calculate_keyword_scores(text)
            return keyword_scores
    
    def process_single_document(self, pdf_path: str) -> List[Dict]:
        """
        Traite un seul document PDF avec gestion des cas spéciaux
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Liste des résultats (une entrée par page)
        """
        logger.info(f"Traitement de {pdf_path}")
        
        try:
            # Étape 1: Conversion PDF -> Images
            images = self.pdf_processor.pdf_to_images(pdf_path)
            logger.info(f"PDF converti en {len(images)} page(s)")
            
            results = []
            
            for page_idx, image in enumerate(images):
                logger.info(f"Traitement page {page_idx + 1}/{len(images)}")
                
                # Étape 2: Prétraitement pour OCR
                if self.config['ocr']['preprocessing']['denoise']:
                    preprocessed = self.pdf_processor.preprocess_for_ocr(image)
                else:
                    preprocessed = image
                
                # Étape 3: Extraction OCR
                ocr_result = self.ocr_extractor.process_document(preprocessed)
                text = ocr_result['corrected_text']
                
                # Étape 4: Extraction features gabarits
                gabarit_features = self.gabarit_detector.extract_all_features(
                    image, text
                )
                gabarit_scores = self.gabarit_detector.classify_by_gabarit(
                    image, text
                )
                
                # Étape 5: Classification CV
                cv_scores = self.extract_cv_features(image, gabarit_features)
                
                # Étape 6: Classification NLP
                nlp_scores = self.extract_nlp_features(text)
                
                # Étape 7: Fusion AMÉLIORÉE avec détection cas spéciaux
                decision = self.fusion.make_decision(
                    cv_scores=cv_scores,
                    nlp_scores=nlp_scores,
                    gabarit_scores=gabarit_scores,
                    gabarit_features=gabarit_features,
                    ocr_info=ocr_result
                )
                
                # Ajouter les métadonnées
                result = {
                    'pdf_path': pdf_path,
                    'page_number': page_idx + 1,
                    'predicted_class': decision['predicted_class'],
                    'confidence': decision['confidence'],
                    'should_review': decision['should_review'],
                    'strategy': decision['strategy'],
                    'ocr_confidence': ocr_result['ocr_confidence'],
                    'text_preview': text[:200] + '...' if len(text) > 200 else text,
                    'gabarit_features': gabarit_features,
                    'all_scores': decision['all_scores']
                }
                
                # Ajouter la raison de rejet si présente
                if 'rejection_reason' in decision:
                    result['rejection_reason'] = decision['rejection_reason']
                
                results.append(result)
                
                # Log avec indication cas spécial
                if decision['predicted_class'] in ['DOCUMENT_VIDE', 'DOCUMENT_ILLISIBLE', 'DOCUMENT_NON_RECONNU']:
                    logger.warning(
                        f"Page {page_idx + 1}: {decision['predicted_class']} "
                        f"({decision.get('rejection_reason', 'N/A')})"
                    )
                else:
                    logger.info(
                        f"Page {page_idx + 1}: {decision['predicted_class']} "
                        f"(conf={decision['confidence']:.3f}, "
                        f"review={decision['should_review']})"
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {pdf_path}: {e}")
            raise
    
    def process_batch(self, pdf_dir: str, output_dir: str = None) -> Dict:
        """
        Traite un dossier de PDFs
        
        Args:
            pdf_dir: Dossier contenant les PDFs
            output_dir: Dossier de sortie pour les résultats
            
        Returns:
            Dictionnaire avec tous les résultats
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Dossier {pdf_dir} non trouvé")
        
        # Lister tous les PDFs
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Trouvé {len(pdf_files)} fichier(s) PDF à traiter")
        
        if len(pdf_files) == 0:
            logger.warning("Aucun fichier PDF trouvé")
            return {}
        
        # Traiter chaque PDF
        all_results = []
        failed_files = []
        
        for pdf_file in tqdm(pdf_files, desc="Traitement des documents"):
            try:
                results = self.process_single_document(str(pdf_file))
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Échec du traitement de {pdf_file}: {e}")
                failed_files.append(str(pdf_file))
        
        # Organiser les résultats par classe (incluant cas spéciaux)
        results_by_class = {cls: [] for cls in self.classes}
        results_by_class['to_review'] = []
        results_by_class['DOCUMENT_VIDE'] = []
        results_by_class['DOCUMENT_ILLISIBLE'] = []
        results_by_class['DOCUMENT_NON_RECONNU'] = []
        
        for result in all_results:
            if result['should_review']:
                results_by_class['to_review'].append(result)
            
            pred_class = result['predicted_class']
            if pred_class in results_by_class:
                results_by_class[pred_class].append(result)
        
        # Calculer les statistiques
        stats = self.fusion.get_statistics([
            {'predicted_class': r['predicted_class'],
             'confidence': r['confidence'],
             'should_review': r['should_review'],
             'strategy': r['strategy']}
            for r in all_results
        ])
        
        # Sauvegarder les résultats si demandé
        if output_dir:
            self.save_results(all_results, results_by_class, stats, output_dir)
        
        return {
            'all_results': all_results,
            'results_by_class': results_by_class,
            'statistics': stats,
            'failed_files': failed_files
        }
    
    def save_results(self, all_results: List[Dict],
                    results_by_class: Dict,
                    stats: Dict,
                    output_dir: str):
        """
        Sauvegarde les résultats
        """
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Fonction de conversion
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Nettoyer et sauvegarder
        clean_results = convert_to_serializable(all_results)
        clean_stats = convert_to_serializable(stats)
        
        with open(output_path / 'all_results.json', 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(clean_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Résultats sauvegardés dans {output_dir}")


if __name__ == "__main__":
    # Test du pipeline
    pipeline = ImprovedDocumentClassificationPipeline()
    print("Pipeline AMÉLIORÉ prêt à l'emploi")
    print("✨ Détection activée: vides, illisibles, hors classes")
