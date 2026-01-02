"""
Pipeline principal de classification de documents
Intègre tous les modules: prétraitement, CV, NLP, gabarits, fusion
"""
import sys
import os
from pathlib import Path
import numpy as np
from typing import List, Dict
import logging
import yaml
from tqdm import tqdm

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.pdf_processor import PDFPreprocessor
from gabarits.detector import GabaritsDetector
from nlp.ocr_extractor import OCRExtractor
from fusion.multimodal_fusion import MultimodalFusion
from utils.offline_manager import OfflineModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentClassificationPipeline:
    """
    Pipeline complet de classification de documents
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le pipeline
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        logger.info("Initialisation du pipeline de classification...")
        
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
            logger.warning("Le pipeline fonctionnera en mode dégradé (gabarits + OCR uniquement)")
            self.cv_model = None
            self.nlp_model = None
            self.nlp_tokenizer = None
        
        self.classes = self.config['classes']
        
        logger.info("Pipeline initialisé avec succès")
    
    def extract_cv_features(self, image: np.ndarray, gabarit_features: Dict) -> Dict[str, float]:
        """
        Extrait les features Computer Vision avec le modèle hybride
        
        Args:
            image: Image en numpy array
            gabarit_features: Features de gabarits
            
        Returns:
            Scores par classe
        """
        try:
            # Utiliser le vrai modèle CV si disponible
            if self.cv_model is not None and hasattr(self.cv_model, 'predict'):
                return self.cv_model.predict(image, gabarit_features)
            else:
                # Fallback: combiner gabarit scores avec une légère randomisation
                logger.warning("Modèle CV non chargé, utilisation des gabarits uniquement")
                gabarit_scores = self.gabarit_detector.classify_by_gabarit(image, "")
                return gabarit_scores
        except Exception as e:
            logger.error(f"Erreur extraction CV features: {e}")
            return {cls: 0.2 for cls in self.classes}
    
    def extract_nlp_features(self, text: str) -> Dict[str, float]:
        """
        Extrait les features NLP avec CamemBERT
        
        Args:
            text: Texte à analyser
            
        Returns:
            Scores par classe
        """
        try:
            # Utiliser le vrai modèle NLP si disponible
            if self.nlp_model is not None and self.nlp_tokenizer is not None:
                # Créer un classificateur NLP temporaire
                from nlp.nlp_model import NLPClassifier
                nlp_classifier = NLPClassifier()
                return nlp_classifier.predict(text)
            else:
                # Fallback: utiliser les scores de mots-clés
                logger.warning("Modèle NLP non chargé, utilisation des mots-clés")
                keyword_scores = self.ocr_extractor.calculate_keyword_scores(text)
                return keyword_scores
        except Exception as e:
            logger.error(f"Erreur extraction NLP features: {e}")
            keyword_scores = self.ocr_extractor.calculate_keyword_scores(text)
            return keyword_scores
    
    def process_single_document(self, pdf_path: str) -> List[Dict]:
        """
        Traite un seul document PDF
        
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
                
                # Étape 7: Fusion multimodale et décision finale
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
                
                results.append(result)
                
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
        
        # Organiser les résultats par classe
        results_by_class = {cls: [] for cls in self.classes}
        results_by_class['to_review'] = []
        
        for result in all_results:
            if result['should_review']:
                results_by_class['to_review'].append(result)
            else:
                results_by_class[result['predicted_class']].append(result)
        
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
        Sauvegarde les résultats dans des fichiers
        
        Args:
            all_results: Liste de tous les résultats
            results_by_class: Résultats organisés par classe
            stats: Statistiques
            output_dir: Dossier de sortie
        """
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder tous les résultats
        with open(output_path / 'all_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder les statistiques
        with open(output_path / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Créer un rapport textuel
        report_lines = [
            "=" * 60,
            "RAPPORT DE CLASSIFICATION DE DOCUMENTS",
            "=" * 60,
            "",
            f"Total de documents traités: {stats['total_documents']}",
            f"Documents à revoir: {stats['to_review']} ({stats['to_review_percentage']:.1f}%)",
            f"Confiance moyenne: {stats['average_confidence']:.3f}",
            "",
            "DISTRIBUTION PAR CLASSE:",
            "-" * 60
        ]
        
        for cls, count in stats['class_distribution'].items():
            percentage = (count / stats['total_documents']) * 100
            report_lines.append(f"  {cls}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "STRATÉGIES UTILISÉES:",
            "-" * 60
        ])
        
        for strategy, count in stats['strategies_used'].items():
            report_lines.append(f"  {strategy}: {count}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        with open(output_path / 'report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Résultats sauvegardés dans {output_dir}")


if __name__ == "__main__":
    # Test du pipeline
    pipeline = DocumentClassificationPipeline()
    print("Pipeline prêt à l'emploi")
