"""
Module de fusion multimodale pour combiner les prédictions CV, NLP et Gabarits
Version améliorée avec gestion des documents vides et hors classes
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusion:
    """
    Système expert de fusion multimodale avec gestion robuste des cas spéciaux
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le système de fusion
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.fusion_config = config['fusion']
        self.classes = config['classes']
        
        # Seuils de confiance
        self.thresholds = self.fusion_config['confidence_thresholds']
        
        # Poids des modèles
        self.weights = self.fusion_config['weights']
        
        # Seuils spécifiques
        self.perfect_agreement = self.fusion_config['perfect_agreement_threshold']
        self.rejection_threshold = self.fusion_config['rejection_threshold']
        
        # NOUVEAU: Seuils pour détection de documents problématiques
        self.min_word_count = 5  # Minimum de mots pour considérer le document valide
        self.min_ocr_confidence = 30  # Confiance OCR minimale (%)
        self.uncertainty_threshold = 0.25  # Si tous les scores < 40%, document non reconnu
        
        logger.info(f"MultimodalFusion initialisé avec {len(self.classes)} classes: {self.classes}")
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalise les scores pour qu'ils somment à 1
        
        Args:
            scores: Dictionnaire de scores
            
        Returns:
            Scores normalisés
        """
        total = sum(scores.values())
        if total == 0:
            return {k: 1.0/len(scores) for k in scores.keys()}
        return {k: v/total for k, v in scores.items()}
    
    def check_perfect_agreement(self, 
                                cv_pred: str, 
                                nlp_pred: str,
                                cv_conf: float,
                                nlp_conf: float) -> Optional[Tuple[str, float]]:
        """
        Vérifie s'il y a un accord parfait entre CV et NLP
        
        Args:
            cv_pred: Prédiction CV
            nlp_pred: Prédiction NLP
            cv_conf: Confiance CV
            nlp_conf: Confiance NLP
            
        Returns:
            Tuple (classe, confiance) si accord parfait, None sinon
        """
        if cv_pred == nlp_pred and cv_conf > self.perfect_agreement and nlp_conf > self.perfect_agreement:
            final_conf = (cv_conf + nlp_conf) / 2.0
            logger.debug(f"Accord parfait: {cv_pred} (conf={final_conf:.3f})")
            return cv_pred, final_conf
        
        return None
    
    def validate_with_gabarits(self, 
                               predicted_class: str,
                               gabarit_scores: Dict[str, float]) -> float:
        """
        Valide une prédiction avec les scores de gabarits
        
        Args:
            predicted_class: Classe prédite
            gabarit_scores: Scores des gabarits
            
        Returns:
            Score de validation (0-1)
        """
        if predicted_class in gabarit_scores:
            return gabarit_scores[predicted_class]
        return 0.0
    
    def apply_business_rules(self,
                            predicted_class: str,
                            gabarit_features: Dict[str, float],
                            ocr_info: Dict) -> bool:
        """
        Applique les règles métier pour valider une prédiction
        
        Args:
            predicted_class: Classe prédite
            gabarit_features: Features des gabarits
            ocr_info: Informations extraites par OCR
            
        Returns:
            True si la prédiction est valide selon les règles métier
        """
        if predicted_class == "piece_identite":
            # Pièce d'identité: Doit avoir une photo ET un format carte
            has_photo = gabarit_features.get('has_photo', 0) > 0.5
            aspect_ratio = gabarit_features.get('aspect_ratio', 0)
            is_card_format = 1.4 < aspect_ratio < 1.8
            
            return has_photo and is_card_format
        
        elif predicted_class == "releve_notes":
            # Relevé de notes: Doit avoir une structure tabulaire OU une densité numérique élevée
            has_table = gabarit_features.get('has_table', 0) > 0.3
            has_numeric = gabarit_features.get('numeric_density', 0) > 0.1
            
            # Vérifier aussi la présence de mots-clés pertinents
            text = ocr_info.get('corrected_text', '').lower()
            has_notes_keywords = any(word in text for word in [
                'relevé', 'notes', 'semestre', 'module', 'moyenne', 
                'coefficient', 'examen', 'ensam', 'cp1', 'cp2'
            ])
            
            return (has_table or has_numeric) and has_notes_keywords
        
        elif predicted_class == "facture":
            # Facture: Doit avoir structure tabulaire ET unités de mesure typiques
            has_table = gabarit_features.get('has_table', 0) > 0.3
            text = ocr_info.get('corrected_text', '').lower()
            
            # Vérifier unités de mesure (électricité, eau) ou mots-clés facture
            has_unit = any(unit in text for unit in [
                'kwh', 'kilowatt', 'm³', 'm3', 'mètre cube'
            ])
            has_facture_keywords = any(word in text for word in [
                'facture', 'montant', 'total', 'dh', 'paiement', 
                'consommation', 'abonnement'
            ])
            
            return has_table and (has_unit or has_facture_keywords)
        
        # Par défaut, accepter la prédiction
        return True
    
    def weighted_fusion(self,
                       cv_scores: Dict[str, float],
                       nlp_scores: Dict[str, float],
                       gabarit_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Fusion pondérée des scores de tous les modèles
        
        Args:
            cv_scores: Scores Computer Vision
            nlp_scores: Scores NLP
            gabarit_scores: Scores Gabarits
            
        Returns:
            Scores fusionnés
        """
        # Normaliser tous les scores
        cv_norm = self.normalize_scores(cv_scores)
        nlp_norm = self.normalize_scores(nlp_scores)
        gabarit_norm = self.normalize_scores(gabarit_scores)
        
        # Fusion pondérée
        fused_scores = {}
        for cls in self.classes:
            cv_score = cv_norm.get(cls, 0) * self.weights['cv']
            nlp_score = nlp_norm.get(cls, 0) * self.weights['nlp']
            gabarit_score = gabarit_norm.get(cls, 0) * self.weights['gabarits']
            
            fused_scores[cls] = cv_score + nlp_score + gabarit_score
        
        return self.normalize_scores(fused_scores)
    
    def make_decision(self,
                     cv_scores: Dict[str, float],
                     nlp_scores: Dict[str, float],
                     gabarit_scores: Dict[str, float],
                     gabarit_features: Dict[str, float],
                     ocr_info: Dict) -> Dict:
        """
        Prend la décision finale de classification avec gestion des cas spéciaux
        
        Args:
            cv_scores: Scores Computer Vision
            nlp_scores: Scores NLP
            gabarit_scores: Scores Gabarits
            gabarit_features: Features structurelles
            ocr_info: Informations OCR
            
        Returns:
            Dictionnaire avec la décision et les métadonnées
        """
        # ==========================================
        # ÉTAPE 0: Détection de PDF vide ou invalide
        # ==========================================
        text = ocr_info.get('corrected_text', '')
        word_count = ocr_info.get('word_count', 0)
        ocr_confidence = ocr_info.get('ocr_confidence', 0)
        
        # Cas 1: PDF complètement vide (moins de 5 mots)
        if word_count < self.min_word_count or len(text.strip()) < 10:
            logger.warning("Document vide ou illisible détecté")
            return {
                'predicted_class': 'DOCUMENT_VIDE',
                'confidence': 0.0,
                'strategy': 'empty_document_detected',
                'should_review': True,
                'rejection_reason': 'Texte insuffisant (moins de 5 mots)',
                'all_scores': {
                    'cv_scores': cv_scores,
                    'nlp_scores': nlp_scores,
                    'gabarit_scores': gabarit_scores
                }
            }
        
        # Cas 2: OCR très faible confiance (document illisible)
        if ocr_confidence < self.min_ocr_confidence:
            logger.warning(f"Document illisible détecté (confiance OCR: {ocr_confidence}%)")
            return {
                'predicted_class': 'DOCUMENT_ILLISIBLE',
                'confidence': ocr_confidence / 100.0,
                'strategy': 'low_ocr_quality',
                'should_review': True,
                'rejection_reason': f'Qualité OCR trop faible ({ocr_confidence}%)',
                'all_scores': {
                    'cv_scores': cv_scores,
                    'nlp_scores': nlp_scores,
                    'gabarit_scores': gabarit_scores
                }
            }
        
        # ==========================================
        # ÉTAPE 1: Classification normale
        # ==========================================
        
        # Extraire les prédictions avec confiance maximale
        cv_pred = max(cv_scores.items(), key=lambda x: x[1])
        nlp_pred = max(nlp_scores.items(), key=lambda x: x[1])
        
        # Stratégie 1: Accord parfait CV + NLP
        perfect_agreement = self.check_perfect_agreement(
            cv_pred[0], nlp_pred[0], cv_pred[1], nlp_pred[1]
        )
        
        if perfect_agreement:
            pred_class, confidence = perfect_agreement
            
            # Valider avec gabarits
            gabarit_validation = self.validate_with_gabarits(pred_class, gabarit_scores)
            
            if gabarit_validation > 0.7:
                return {
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'strategy': 'perfect_agreement_validated',
                    'should_review': False,
                    'all_scores': {
                        'cv_scores': cv_scores,
                        'nlp_scores': nlp_scores,
                        'gabarit_scores': gabarit_scores
                    }
                }
        
        # Stratégie 2: CV fort + Validation gabarits
        if cv_pred[1] > self.thresholds['high']:
            gabarit_validation = self.validate_with_gabarits(cv_pred[0], gabarit_scores)
            
            if gabarit_validation > 0.7:
                return {
                    'predicted_class': cv_pred[0],
                    'confidence': cv_pred[1],
                    'strategy': 'cv_strong_gabarit_validated',
                    'should_review': False,
                    'all_scores': {
                        'cv_scores': cv_scores,
                        'nlp_scores': nlp_scores,
                        'gabarit_scores': gabarit_scores
                    }
                }
        
        # Stratégie 3: NLP fort + Motifs textuels
        if nlp_pred[1] > self.thresholds['high']:
            keyword_score = ocr_info.get('keyword_scores', {}).get(nlp_pred[0], 0)
            
            if keyword_score > 0.6:
                return {
                    'predicted_class': nlp_pred[0],
                    'confidence': nlp_pred[1],
                    'strategy': 'nlp_strong_keywords_validated',
                    'should_review': False,
                    'all_scores': {
                        'cv_scores': cv_scores,
                        'nlp_scores': nlp_scores,
                        'gabarit_scores': gabarit_scores
                    }
                }
        
        # Stratégie 4: Fusion pondérée
        fused_scores = self.weighted_fusion(cv_scores, nlp_scores, gabarit_scores)
        final_pred = max(fused_scores.items(), key=lambda x: x[1])
        
        # ==========================================
        # ÉTAPE 2: Détection de document hors classes
        # ==========================================
        
        # Si tous les scores fusionnés sont très faibles → document non reconnu
        if final_pred[1] < self.uncertainty_threshold:
            logger.warning(f"Document non reconnu (confiance maximale: {final_pred[1]:.3f})")
            return {
                'predicted_class': 'DOCUMENT_NON_RECONNU',
                'confidence': final_pred[1],
                'strategy': 'high_uncertainty',
                'should_review': True,
                'rejection_reason': f'Aucune classe ne correspond (confiance max: {final_pred[1]:.1%})',
                'all_scores': {
                    'cv_scores': cv_scores,
                    'nlp_scores': nlp_scores,
                    'gabarit_scores': gabarit_scores,
                    'fused_scores': fused_scores
                }
            }
        
        # Appliquer les règles métier
        business_rule_valid = self.apply_business_rules(
            final_pred[0], gabarit_features, ocr_info
        )
        
        # Décider si le document doit être revu manuellement
        should_review = (
            final_pred[1] < self.rejection_threshold or
            not business_rule_valid
        )
        
        return {
            'predicted_class': final_pred[0],
            'confidence': final_pred[1],
            'strategy': 'weighted_fusion',
            'should_review': should_review,
            'business_rule_valid': business_rule_valid,
            'all_scores': {
                'cv_scores': cv_scores,
                'nlp_scores': nlp_scores,
                'gabarit_scores': gabarit_scores,
                'fused_scores': fused_scores
            }
        }
    
    def batch_classify(self, predictions_list: List[Dict]) -> List[Dict]:
        """
        Classifie un batch de documents
        
        Args:
            predictions_list: Liste de dictionnaires de prédictions
            
        Returns:
            Liste des décisions
        """
        results = []
        
        for pred in predictions_list:
            decision = self.make_decision(
                pred['cv_scores'],
                pred['nlp_scores'],
                pred['gabarit_scores'],
                pred['gabarit_features'],
                pred['ocr_info']
            )
            results.append(decision)
        
        return results
    
    def get_statistics(self, decisions: List[Dict]) -> Dict:
        """
        Calcule des statistiques sur un ensemble de décisions
        
        Args:
            decisions: Liste des décisions
            
        Returns:
            Statistiques
        """
        total = len(decisions)
        if total == 0:
            return {}
        
        # Compter les décisions par classe (incluant cas spéciaux)
        class_counts = {cls: 0 for cls in self.classes}
        class_counts['DOCUMENT_VIDE'] = 0
        class_counts['DOCUMENT_ILLISIBLE'] = 0
        class_counts['DOCUMENT_NON_RECONNU'] = 0
        
        for decision in decisions:
            pred_class = decision['predicted_class']
            if pred_class in class_counts:
                class_counts[pred_class] += 1
            else:
                class_counts[pred_class] = 1
        
        # Compter les documents à revoir
        to_review = sum(1 for d in decisions if d['should_review'])
        
        # Confiance moyenne (exclure les cas spéciaux)
        valid_confidences = [
            d['confidence'] for d in decisions 
            if d['predicted_class'] not in ['DOCUMENT_VIDE', 'DOCUMENT_ILLISIBLE', 'DOCUMENT_NON_RECONNU']
        ]
        avg_confidence = np.mean(valid_confidences) if valid_confidences else 0
        
        # Stratégies utilisées
        strategies = {}
        for decision in decisions:
            strategy = decision['strategy']
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        return {
            'total_documents': total,
            'class_distribution': class_counts,
            'to_review': to_review,
            'to_review_percentage': (to_review / total) * 100,
            'average_confidence': avg_confidence,
            'strategies_used': strategies
        }


if __name__ == "__main__":
    # Test de la fusion
    fusion = MultimodalFusion()
    print(f"MultimodalFusion prêt avec {len(fusion.classes)} classes: {fusion.classes}")
