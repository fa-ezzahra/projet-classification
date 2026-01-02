"""
Module de prétraitement des documents PDF
Conversion PDF -> Images et amélioration de la qualité
"""
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """
    Classe pour le prétraitement des documents PDF
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialise le préprocesseur
        
        Args:
            dpi: Résolution pour la conversion PDF -> Image
        """
        self.dpi = dpi
        logger.info(f"PDFPreprocessor initialisé avec DPI={dpi}")
    
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convertit un PDF en liste d'images
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Liste d'images au format numpy array
        """
        try:
            # Convertir PDF en images PIL
            pil_images = convert_from_path(pdf_path, dpi=self.dpi)
            
            # Convertir en numpy arrays
            images = []
            for pil_img in pil_images:
                img_array = np.array(pil_img)
                # Convertir RGB en BGR (format OpenCV)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_array)
            
            logger.info(f"PDF converti en {len(images)} image(s)")
            return images
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion PDF: {e}")
            raise
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un débruitage à l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image débruitée
        """
        # Débruitage avec filtre Non-Local Means
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image avec contraste amélioré
        """
        # Convertir en LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Appliquer CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Recombiner les canaux
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Corrige l'inclinaison de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image redressée
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Inverser les couleurs
        gray = cv2.bitwise_not(gray)
        
        # Détecter les bords
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Détecter les lignes avec transformée de Hough
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            return image
        
        # Calculer l'angle moyen
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        
        # Appliquer la rotation
        if abs(median_angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Binarise l'image (noir et blanc)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image binarisée
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def remove_borders(self, image: np.ndarray, border_size: int = 20) -> np.ndarray:
        """
        Supprime les bordures de l'image
        
        Args:
            image: Image en numpy array
            border_size: Taille des bordures à supprimer (en pixels)
            
        Returns:
            Image sans bordures
        """
        h, w = image.shape[:2]
        return image[border_size:h-border_size, border_size:w-border_size]
    
    def preprocess_for_ocr(self, image: np.ndarray,
                          denoise: bool = True,
                          enhance: bool = True,
                          deskew: bool = True) -> np.ndarray:
        """
        Pipeline complet de prétraitement pour l'OCR
        
        Args:
            image: Image en numpy array
            denoise: Appliquer le débruitage
            enhance: Améliorer le contraste
            deskew: Corriger l'inclinaison
            
        Returns:
            Image prétraitée
        """
        processed = image.copy()
        
        if denoise:
            processed = self.denoise_image(processed)
        
        if enhance:
            processed = self.enhance_contrast(processed)
        
        if deskew:
            processed = self.deskew_image(processed)
        
        return processed
    
    def resize_image(self, image: np.ndarray,
                    target_size: Tuple[int, int]) -> np.ndarray:
        """
        Redimensionne l'image
        
        Args:
            image: Image en numpy array
            target_size: Taille cible (width, height)
            
        Returns:
            Image redimensionnée
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def process_document(self, pdf_path: str,
                        preprocess_for_ocr: bool = True,
                        save_processed: bool = False,
                        output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Traite un document PDF complet
        
        Args:
            pdf_path: Chemin vers le PDF
            preprocess_for_ocr: Appliquer le prétraitement pour OCR
            save_processed: Sauvegarder les images traitées
            output_dir: Dossier de sortie pour les images
            
        Returns:
            Liste des images traitées
        """
        # Convertir PDF en images
        images = self.pdf_to_images(pdf_path)
        
        # Prétraiter chaque image
        processed_images = []
        for idx, img in enumerate(images):
            if preprocess_for_ocr:
                processed = self.preprocess_for_ocr(img)
            else:
                processed = img
            
            processed_images.append(processed)
            
            # Sauvegarder si demandé
            if save_processed and output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                pdf_name = Path(pdf_path).stem
                img_path = output_path / f"{pdf_name}_page_{idx+1}.png"
                cv2.imwrite(str(img_path), processed)
        
        logger.info(f"Document traité: {len(processed_images)} page(s)")
        return processed_images


if __name__ == "__main__":
    # Test du préprocesseur
    preprocessor = PDFPreprocessor(dpi=300)
    print("PDFPreprocessor prêt à l'emploi")
