#!/usr/bin/env python3
"""
Script d'augmentation UNIQUEMENT pour releve_notes
N'affecte pas piece_identite et facture déjà existants
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import random
from PIL import Image, ImageEnhance
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentor:
    """Classe pour l'augmentation de données d'images de documents"""
    
    def __init__(self):
        self.augmentation_techniques = [
            self.rotate,
            self.perspective_transform,
            self.add_noise,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_blur,
            self.add_shadow,
            self.scale,
            self.add_compression_artifacts
        ]
    
    def rotate(self, img, angle_range=(-15, 15)):
        """Rotation aléatoire"""
        angle = random.uniform(*angle_range)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def perspective_transform(self, img):
        """Transformation de perspective (simule scan de travers)"""
        h, w = img.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = int(min(w, h) * 0.05)
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)]
        ])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def add_noise(self, img):
        """Ajoute du bruit gaussien"""
        noise = np.random.normal(0, random.randint(5, 15), img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def adjust_brightness(self, img):
        """Ajuste la luminosité"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        factor = random.uniform(0.7, 1.3)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def adjust_contrast(self, img):
        """Ajuste le contraste"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        factor = random.uniform(0.8, 1.4)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def add_blur(self, img):
        """Ajoute un flou léger"""
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def add_shadow(self, img):
        """Ajoute une ombre (simule éclairage non uniforme)"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            for i in range(h):
                mask[i, :] = i / h
        elif direction == 'vertical':
            for j in range(w):
                mask[:, j] = j / w
        else:
            for i in range(h):
                for j in range(w):
                    mask[i, j] = (i + j) / (h + w)
        
        if random.random() < 0.5:
            mask = 1 - mask
        
        factor = random.uniform(0.6, 0.9)
        mask = mask * (1 - factor) + factor
        mask = np.expand_dims(mask, axis=2)
        
        return np.clip(img * mask, 0, 255).astype(np.uint8)
    
    def scale(self, img):
        """Zoom/dézoom"""
        h, w = img.shape[:2]
        scale_factor = random.uniform(0.9, 1.1)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = cv2.resize(img, (new_w, new_h))
        
        if scale_factor > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h+h, start_w:start_w+w]
        else:
            result = np.ones((h, w, 3), dtype=np.uint8) * 255
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            result[start_h:start_h+new_h, start_w:start_w+new_w] = resized
            return result
    
    def add_compression_artifacts(self, img):
        """Simule compression JPEG"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 95)]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def augment(self, img, num_transforms=3):
        """Applique plusieurs transformations aléatoires"""
        transforms = random.sample(self.augmentation_techniques, num_transforms)
        result = img.copy()
        for transform in transforms:
            try:
                result = transform(result)
            except Exception as e:
                logger.warning(f"Erreur dans transformation {transform.__name__}: {e}")
        return result


def augment_releve_notes(target_per_class=700):
    """
    Augmente UNIQUEMENT la classe releve_notes
    """
    logger.info("="*60)
    logger.info("AUGMENTATION DE releve_notes UNIQUEMENT")
    logger.info("="*60)
    
    # Chemins
    source_dir = Path("data/organized/releve_notes")
    output_train = Path("data/augmented/train/releve_notes")
    output_val = Path("data/augmented/val/releve_notes")
    output_test = Path("data/augmented/test/releve_notes")
    
    # Créer les dossiers de sortie
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)
    
    # Charger les images originales
    logger.info(f"\nChargement depuis {source_dir}...")
    image_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    
    if not image_files:
        logger.error(f"❌ Aucune image trouvée dans {source_dir}")
        logger.info("\nVérifiez que les PDFs ont été convertis avec convert_releve_notes.py")
        return False
    
    logger.info(f"✓ {len(image_files)} images trouvées")
    
    # Charger les images
    original_images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            original_images.append(img)
            logger.info(f"  ✓ Chargé: {img_file.name}")
    
    if not original_images:
        logger.error("❌ Aucune image valide chargée")
        return False
    
    logger.info(f"\n✓ {len(original_images)} images originales chargées")
    
    # Calculer les augmentations
    num_originals = len(original_images)
    augmentations_per_original = target_per_class // num_originals
    
    logger.info(f"📊 Génération de {augmentations_per_original} variations par image...")
    logger.info(f"   Cible totale: ~{target_per_class} images")
    
    # Créer l'augmenteur
    augmentor = DataAugmentor()
    all_images = []
    
    # Ajouter les originales
    all_images.extend(original_images)
    
    # Générer les augmentations
    logger.info("\n🔄 Génération des variations...")
    for orig_img in tqdm(original_images, desc="Augmentation"):
        for _ in range(augmentations_per_original):
            try:
                aug_img = augmentor.augment(orig_img, num_transforms=random.randint(2, 4))
                all_images.append(aug_img)
            except Exception as e:
                logger.error(f"Erreur augmentation: {e}")
    
    logger.info(f"\n✓ {len(all_images)} images générées au total")
    
    # Mélanger
    random.shuffle(all_images)
    
    # Split train/val/test (70/15/15)
    n_train = int(len(all_images) * 0.70)
    n_val = int(len(all_images) * 0.15)
    
    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train+n_val]
    test_imgs = all_images[n_train+n_val:]
    
    # Sauvegarder
    logger.info("\n💾 Sauvegarde des images...")
    
    for i, img in enumerate(tqdm(train_imgs, desc="Train")):
        cv2.imwrite(str(output_train / f"releve_notes_{i:05d}.png"), img)
    
    for i, img in enumerate(tqdm(val_imgs, desc="Val")):
        cv2.imwrite(str(output_val / f"releve_notes_{i:05d}.png"), img)
    
    for i, img in enumerate(tqdm(test_imgs, desc="Test")):
        cv2.imwrite(str(output_test / f"releve_notes_{i:05d}.png"), img)
    
    # Résumé
    logger.info("\n" + "="*60)
    logger.info("✅ AUGMENTATION TERMINÉE - releve_notes")
    logger.info("="*60)
    logger.info(f"Images originales: {num_originals}")
    logger.info(f"Train: {len(train_imgs)}")
    logger.info(f"Val: {len(val_imgs)}")
    logger.info(f"Test: {len(test_imgs)}")
    logger.info(f"Total généré: {len(all_images)}")
    logger.info("="*60)
    
    # Vérifier toutes les classes maintenant
    logger.info("\n📊 RÉSUMÉ COMPLET DU DATASET:")
    for cls in ["piece_identite", "facture", "releve_notes"]:
        train_dir = Path(f"data/augmented/train/{cls}")
        if train_dir.exists():
            count = len(list(train_dir.glob("*.png")))
            logger.info(f"  {cls}: {count} images")
        else:
            logger.info(f"  {cls}: 0 images (dossier non trouvé)")
    
    logger.info("\n✅ Prêt pour l'entraînement !")
    logger.info("Lancez: python train_models.py --model cv --data-dir data/augmented/train --epochs 10 --batch-size 4")
    
    return True


if __name__ == "__main__":
    import sys
    success = augment_releve_notes(target_per_class=700)
    sys.exit(0 if success else 1)
