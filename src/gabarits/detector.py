"""
Système de détection de gabarits structurels pour la classification de documents
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GabaritsDetector:
    """
    Détecteur de features structurelles pour identifier les types de documents
    """
    
    def __init__(self):
        """
        Initialise le détecteur de gabarits
        """
        # Chargement du détecteur de visages (pour les pièces d'identité)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info("GabaritsDetector initialisé")
    
    def detect_photo_zone(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une zone photo (pour cartes d'identité)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Tuple (présence photo, score de confiance)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Détection de visages
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        has_photo = len(faces) > 0
        confidence = min(len(faces) * 0.5, 1.0)  # Score basé sur le nombre de visages
        
        return has_photo, confidence
    
    def calculate_aspect_ratio(self, image: np.ndarray) -> float:
        """
        Calcule le ratio d'aspect de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Ratio largeur/hauteur
        """
        h, w = image.shape[:2]
        return w / h if h > 0 else 0
    
    def detect_table_structure(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une structure tabulaire
        
        Args:
            image: Image en numpy array
            
        Returns:
            Tuple (présence table, score de confiance)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Détection des bords
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Détection de lignes horizontales
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.HoughLinesP(
            detect_horizontal, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10
        )
        
        # Détection de lignes verticales
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.HoughLinesP(
            detect_vertical, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10
        )
        
        h_count = len(horizontal_lines) if horizontal_lines is not None else 0
        v_count = len(vertical_lines) if vertical_lines is not None else 0
        
        # Une table a généralement plusieurs lignes horizontales et verticales
        has_table = h_count >= 3 and v_count >= 2
        confidence = min((h_count + v_count) / 20.0, 1.0)
        
        return has_table, confidence
    
    def calculate_text_density(self, image: np.ndarray) -> float:
        """
        Calcule la densité de texte dans l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Score de densité de texte (0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Binarisation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculer le ratio de pixels noirs (texte) vs blancs
        text_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        density = text_pixels / total_pixels if total_pixels > 0 else 0
        
        return density
    
    def detect_numeric_content(self, image: np.ndarray, ocr_text: str = "") -> float:
        """
        Détecte la présence de contenu numérique
        
        Args:
            image: Image en numpy array
            ocr_text: Texte extrait par OCR (optionnel)
            
        Returns:
            Score de contenu numérique (0-1)
        """
        if ocr_text:
            # Compter les chiffres dans le texte
            digit_count = sum(c.isdigit() for c in ocr_text)
            total_chars = len([c for c in ocr_text if c.isalnum()])
            
            if total_chars > 0:
                return digit_count / total_chars
        
        return 0.0
    
    def detect_signature_zone(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une zone de signature
        
        Args:
            image: Image en numpy array
            
        Returns:
            Tuple (présence signature, score de confiance)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Prendre la partie inférieure de l'image (où se trouvent généralement les signatures)
        h = gray.shape[0]
        bottom_third = gray[int(h*0.66):, :]
        
        # Détecter les traits (signature manuscrite)
        edges = cv2.Canny(bottom_third, 50, 150)
        
        # Calculer la densité de contours dans cette zone
        contour_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        has_signature = contour_density > 0.02  # Seuil empirique
        confidence = min(contour_density / 0.05, 1.0)
        
        return has_signature, confidence
    
    def extract_all_features(self, image: np.ndarray, ocr_text: str = "") -> Dict[str, float]:
        """
        Extrait toutes les features structurelles d'une image
        
        Args:
            image: Image en numpy array
            ocr_text: Texte OCR (optionnel)
            
        Returns:
            Dictionnaire de features
        """
        features = {}
        
        # Ratio d'aspect
        features['aspect_ratio'] = self.calculate_aspect_ratio(image)
        
        # Présence de photo
        has_photo, photo_conf = self.detect_photo_zone(image)
        features['has_photo'] = 1.0 if has_photo else 0.0
        features['photo_confidence'] = photo_conf
        
        # Structure tabulaire
        has_table, table_conf = self.detect_table_structure(image)
        features['has_table'] = 1.0 if has_table else 0.0
        features['table_confidence'] = table_conf
        
        # Densité de texte
        features['text_density'] = self.calculate_text_density(image)
        
        # Contenu numérique
        features['numeric_density'] = self.detect_numeric_content(image, ocr_text)
        
        # Zone de signature
        has_signature, sig_conf = self.detect_signature_zone(image)
        features['has_signature'] = 1.0 if has_signature else 0.0
        features['signature_confidence'] = sig_conf
        
        return features
    
    def match_gabarit(self, features: Dict[str, float], document_type: str) -> float:
        """
        Calcule le score de correspondance avec un gabarit spécifique
        
        Args:
            features: Dictionnaire de features extraites
            document_type: Type de document à matcher
            
        Returns:
            Score de correspondance (0-1)
        """
        score = 0.0
        
        if document_type == "piece_identite":
            # Carte d'identité: format carte + photo + densité texte moyenne
            score += 0.4 if 1.4 < features['aspect_ratio'] < 1.8 else 0.0
            score += 0.4 * features['photo_confidence']
            score += 0.2 if 0.2 < features['text_density'] < 0.5 else 0.0
        
        elif document_type == "releve_notes":
            # Relevé bancaire: structure table + haute densité numérique
            score += 0.5 * features['table_confidence']
            score += 0.3 * features['numeric_density']
            score += 0.2 if features['text_density'] > 0.3 else 0.0
        
        elif document_type in ["facture"]:
            # Factures: structure table + densité numérique + texte structuré
            score += 0.4 * features['table_confidence']
            score += 0.3 * features['numeric_density']
            score += 0.3 if 0.2 < features['text_density'] < 0.6 else 0.0
    
        
        return min(score, 1.0)
    
    def classify_by_gabarit(self, image: np.ndarray, 
                           ocr_text: str = "") -> Dict[str, float]:
        """
        Classifie un document basé sur ses gabarits
        
        Args:
            image: Image en numpy array
            ocr_text: Texte OCR (optionnel)
            
        Returns:
            Dictionnaire des scores par classe
        """
        # Extraire les features
        features = self.extract_all_features(image, ocr_text)
        
        # Calculer les scores pour chaque type de document
        classes = [
            "piece_identite",
            "releve_notes",
            "facture"       
        ]
        
        scores = {}
        for doc_type in classes:
            scores[doc_type] = self.match_gabarit(features, doc_type)
        
        return scores


if __name__ == "__main__":
    # Test du détecteur
    detector = GabaritsDetector()
    print("GabaritsDetector prêt à l'emploi")
