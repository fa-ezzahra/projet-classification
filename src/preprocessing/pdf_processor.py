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
import os
import platform

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
        self.poppler_path = self._detect_poppler()
        logger.info(f"PDFPreprocessor initialisé avec DPI={dpi}")
        
        if self.poppler_path:
            logger.info(f"Poppler détecté à: {self.poppler_path}")
        elif platform.system() == 'Windows':
            logger.warning("=" * 70)
            logger.warning("ATTENTION: Poppler non détecté sur Windows!")
            logger.warning("La conversion PDF ne fonctionnera pas sans Poppler.")
            logger.warning("")
            logger.warning("Pour installer Poppler:")
            logger.warning("1. Téléchargez depuis:")
            logger.warning("   https://github.com/oschwartz10612/poppler-windows/releases/")
            logger.warning("2. Extrayez dans C:\\poppler")
            logger.warning("3. Ajoutez C:\\poppler\\Library\\bin au PATH système")
            logger.warning("   OU placez le dossier dans le répertoire du projet")
            logger.warning("=" * 70)
    
    def _detect_poppler(self) -> Optional[str]:
        """
        Détecte automatiquement l'emplacement de Poppler sur Windows
        
        Returns:
            Chemin vers Poppler ou None
        """
        if platform.system() != 'Windows':
            return None
        
        # Chemins possibles pour Poppler
        possible_paths = [
            r'C:\Program Files\poppler-24.08.0\Library\bin',
            r'C:\Program Files\poppler\Library\bin',
            r'C:\poppler-24.08.0\Library\bin',
            r'C:\poppler\Library\bin',
            os.path.join(os.getcwd(), 'poppler-24.08.0', 'Library', 'bin'),
            os.path.join(os.getcwd(), 'poppler', 'Library', 'bin'),
            os.path.join(os.path.dirname(__file__), '..', 'poppler-24.08.0', 'Library', 'bin'),
            os.path.join(os.path.dirname(__file__), '..', 'poppler', 'Library', 'bin'),
        ]
        
        # Chercher pdftoppm.exe
        for path in possible_paths:
            if os.path.exists(path):
                pdftoppm = os.path.join(path, 'pdftoppm.exe')
                if os.path.exists(pdftoppm):
                    return path
        
        return None
    
    def pdf_to_images(self, pdf_path: str, timeout: int = 60) -> List[np.ndarray]:
        """
        Convertit un PDF en liste d'images
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            timeout: Timeout en secondes (défaut: 60s)
            
        Returns:
            Liste d'images au format numpy array
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Fichier PDF non trouvé: {pdf_path}")
        
        try:
            logger.info(f"Début conversion PDF: {pdf_path}")
            logger.info(f"DPI: {self.dpi}, Timeout: {timeout}s")
            
            # Paramètres pour convert_from_path
            convert_params = {
                'dpi': self.dpi,
                'timeout': timeout,
                'fmt': 'jpeg',  # Format plus rapide que PNG
                'thread_count': 1  # Éviter les problèmes de thread
            }
            
            # Ajouter poppler_path si sur Windows
            if self.poppler_path:
                convert_params['poppler_path'] = self.poppler_path
            elif platform.system() == 'Windows':
                logger.error("=" * 70)
                logger.error("ERREUR: Poppler n'est pas installé!")
                logger.error("")
                logger.error("Solution rapide:")
                logger.error("1. Téléchargez: https://github.com/oschwartz10612/poppler-windows/releases/")
                logger.error("2. Extrayez dans le dossier du projet sous le nom 'poppler'")
                logger.error("3. Relancez le programme")
                logger.error("=" * 70)
                raise FileNotFoundError(
                    "Poppler requis mais non trouvé. "
                    "Téléchargez depuis: https://github.com/oschwartz10612/poppler-windows/releases/"
                )
            
            # Convertir PDF en images PIL
            logger.info("Conversion en cours...")
            pil_images = convert_from_path(pdf_path, **convert_params)
            logger.info(f"Conversion réussie: {len(pil_images)} page(s)")
            
            # Convertir en numpy arrays
            images = []
            for idx, pil_img in enumerate(pil_images):
                logger.debug(f"Traitement page {idx+1}/{len(pil_images)}")
                
                img_array = np.array(pil_img)
                
                # Convertir RGB en BGR (format OpenCV)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                images.append(img_array)
            
            logger.info(f"PDF converti en {len(images)} image(s)")
            return images
            
        except FileNotFoundError as e:
            logger.error(f"Fichier non trouvé: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion PDF: {type(e).__name__}: {e}")
            logger.error(f"Fichier: {pdf_path}")
            
            # Diagnostic plus détaillé
            if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
                logger.error("")
                logger.error("Erreur liée à Poppler. Vérifiez:")
                logger.error("1. Poppler est installé")
                logger.error("2. Le chemin vers Poppler est correct")
                logger.error("3. pdftoppm.exe existe dans le dossier bin")
            
            raise
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un débruitage à l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image débruitée
        """
        try:
            # Débruitage avec filtre Non-Local Means
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            return denoised
        except Exception as e:
            logger.warning(f"Erreur lors du débruitage: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image avec contraste amélioré
        """
        try:
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
        except Exception as e:
            logger.warning(f"Erreur lors de l'amélioration du contraste: {e}")
            return image
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Corrige l'inclinaison de l'image
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image redressée
        """
        try:
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
        except Exception as e:
            logger.warning(f"Erreur lors de la correction d'inclinaison: {e}")
            return image
    
    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Binarise l'image (noir et blanc)
        
        Args:
            image: Image en numpy array
            
        Returns:
            Image binarisée
        """
        try:
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
        except Exception as e:
            logger.warning(f"Erreur lors de la binarisation: {e}")
            return image
    
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
        if h > 2*border_size and w > 2*border_size:
            return image[border_size:h-border_size, border_size:w-border_size]
        return image
    
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
            logger.info(f"Prétraitement page {idx+1}/{len(images)}")
            
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
    
    # Test de détection Poppler
    if platform.system() == 'Windows':
        if preprocessor.poppler_path:
            print(f"✓ Poppler détecté: {preprocessor.poppler_path}")
        else:
            print("✗ Poppler non détecté - la conversion PDF ne fonctionnera pas")