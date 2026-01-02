"""
Module OCR pour l'extraction de texte à partir d'images
"""
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Extracteur de texte utilisant Tesseract OCR
    """
    
    def __init__(self, lang: str = 'fra', psm: int = 3, oem: int = 3):
        """
        Initialise l'extracteur OCR
        
        Args:
            lang: Langue pour l'OCR (fra pour français)
            psm: Page Segmentation Mode
            oem: OCR Engine Mode
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        
        # Configuration Tesseract
        self.config = f'--oem {oem} --psm {psm}'
        
        logger.info(f"OCRExtractor initialisé (lang={lang}, psm={psm}, oem={oem})")
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extrait le texte d'une image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Texte extrait
        """
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config
            )
            return text.strip()
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction OCR: {e}")
            return ""
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Dict:
        """
        Extrait le texte avec les scores de confiance
        
        Args:
            image: Image en numpy array
            
        Returns:
            Dictionnaire contenant le texte et les scores
        """
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filtrer les mots avec une confiance > 0
            words = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:
                    words.append(data['text'][i])
                    confidences.append(int(conf))
            
            text = ' '.join(words)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'word_count': len(words)
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction OCR avec confiance: {e}")
            return {'text': '', 'confidence': 0, 'word_count': 0}
    
    def correct_common_errors(self, text: str) -> str:
        """
        Corrige les erreurs courantes d'OCR
        
        Args:
            text: Texte à corriger
            
        Returns:
            Texte corrigé
        """
        # Corrections courantes
        corrections = {
            r'\bl\b': 'I',  # l minuscule -> I majuscule
            r'0': 'O',      # 0 -> O dans certains contextes
            r'\bII\b': 'H', # II -> H
        }
        
        corrected = text
        for pattern, replacement in corrections.items():
            corrected = re.sub(pattern, replacement, corrected)
        
        return corrected
    
    def extract_keywords_by_category(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les mots-clés par catégorie de document
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire des mots-clés trouvés par catégorie
        """
        text_lower = text.lower()
        
        keywords = {
            'piece_identite': [
                'identité', 'nationale', 'numéro', 'cnie', 'carte',
                'naissance', 'nationalité', 'marocaine', 'validité',
                'émission', 'prénom', 'nom'
            ],
            'releve_bancaire': [
                'banque', 'compte', 'solde', 'débit', 'crédit',
                'opération', 'mouvement', 'ancien', 'nouveau',
                'rib', 'iban'
            ],
            'facture_electricite': [
                'électricité', 'kwh', 'kilowatt', 'puissance',
                'consommation', 'abonnement', 'one', 'radem',
                'lydec', 'redal', 'compteur'
            ],
            'facture_eau': [
                'eau', 'm³', 'mètre cube', 'consommation',
                'compteur', 'abonnement', 'potable', 'index',
                'facture eau'
            ],
            'document_employeur': [
                'salaire', 'employeur', 'embauche', 'cotisation',
                'bulletin', 'paie', 'attestation', 'travail',
                'employé', 'cnss', 'imposable'
            ]
        }
        
        found_keywords = {}
        for category, words in keywords.items():
            found = [word for word in words if word in text_lower]
            found_keywords[category] = found
        
        return found_keywords
    
    def calculate_keyword_scores(self, text: str) -> Dict[str, float]:
        """
        Calcule des scores basés sur la présence de mots-clés
        
        Args:
            text: Texte à analyser
            
        Returns:
            Scores par catégorie (0-1)
        """
        found_keywords = self.extract_keywords_by_category(text)
        
        scores = {}
        for category, keywords in found_keywords.items():
            # Score basé sur le nombre de mots-clés trouvés
            score = min(len(keywords) / 5.0, 1.0)  # Normaliser sur 5 mots-clés
            scores[category] = score
        
        return scores
    
    def extract_structured_info(self, text: str, doc_type: str) -> Dict:
        """
        Extrait des informations structurées selon le type de document
        
        Args:
            text: Texte à analyser
            doc_type: Type de document
            
        Returns:
            Dictionnaire d'informations extraites
        """
        info = {}
        
        if doc_type == "piece_identite":
            # Extraire numéro CNIE (format: 2 lettres + 6 chiffres)
            cnie_pattern = r'\b[A-Z]{2}\d{6}\b'
            cnie_match = re.search(cnie_pattern, text)
            if cnie_match:
                info['cnie_number'] = cnie_match.group()
            
            # Extraire dates (format: DD/MM/YYYY)
            date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
            dates = re.findall(date_pattern, text)
            if dates:
                info['dates'] = dates
        
        elif doc_type == "releve_bancaire":
            # Extraire montants (format: XXXX.XX DH ou XXXX,XX)
            amount_pattern = r'\b\d{1,10}[.,]\d{2}\b'
            amounts = re.findall(amount_pattern, text)
            if amounts:
                info['amounts'] = amounts
            
            # Extraire RIB
            rib_pattern = r'\b\d{24}\b'
            rib_match = re.search(rib_pattern, text)
            if rib_match:
                info['rib'] = rib_match.group()
        
        elif doc_type in ["facture_electricite", "facture_eau"]:
            # Extraire consommation
            if doc_type == "facture_electricite":
                consumption_pattern = r'(\d+)\s*kwh'
            else:
                consumption_pattern = r'(\d+)\s*m[³3]'
            
            consumption = re.findall(consumption_pattern, text.lower())
            if consumption:
                info['consumption'] = consumption
            
            # Extraire montant total
            total_pattern = r'total[:\s]*(\d+[.,]\d{2})'
            total_match = re.search(total_pattern, text.lower())
            if total_match:
                info['total_amount'] = total_match.group(1)
        
        elif doc_type == "document_employeur":
            # Extraire salaire
            salary_pattern = r'salaire[:\s]*(\d+[.,]\d{2})'
            salary_match = re.search(salary_pattern, text.lower())
            if salary_match:
                info['salary'] = salary_match.group(1)
            
            # Extraire numéro CNSS
            cnss_pattern = r'\b\d{9,10}\b'
            cnss_match = re.search(cnss_pattern, text)
            if cnss_match:
                info['cnss_number'] = cnss_match.group()
        
        return info
    
    def process_document(self, image: np.ndarray) -> Dict:
        """
        Traitement complet d'un document
        
        Args:
            image: Image en numpy array
            
        Returns:
            Dictionnaire complet avec texte, scores et infos structurées
        """
        # Extraction basique
        result = self.extract_text_with_confidence(image)
        text = result['text']
        
        # Correction des erreurs
        corrected_text = self.correct_common_errors(text)
        
        # Calcul des scores par catégorie
        keyword_scores = self.calculate_keyword_scores(corrected_text)
        
        return {
            'raw_text': text,
            'corrected_text': corrected_text,
            'ocr_confidence': result['confidence'],
            'word_count': result['word_count'],
            'keyword_scores': keyword_scores,
            'predicted_class': max(keyword_scores.items(), key=lambda x: x[1])[0] if keyword_scores else None
        }


if __name__ == "__main__":
    # Test de l'extracteur
    ocr = OCRExtractor()
    print("OCRExtractor prêt à l'emploi")
