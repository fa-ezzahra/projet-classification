"""
Script de préparation et d'augmentation du dataset
Génère 1000+ images par classe à partir de quelques exemples
"""
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
from pdf2image import convert_from_path
import random
from PIL import Image, ImageEnhance, ImageFilter
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
        
        # Points source
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Points destination avec perturbation aléatoire
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
        
        # Créer un masque d'ombre
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Gradient aléatoire
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
        
        # Inverser aléatoirement
        if random.random() < 0.5:
            mask = 1 - mask
        
        # Appliquer l'ombre
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
        
        # Recadrer ou remplir pour revenir à la taille originale
        if scale_factor > 1:
            # Recadrer
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h+h, start_w:start_w+w]
        else:
            # Remplir
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
        # Choisir aléatoirement des transformations
        transforms = random.sample(self.augmentation_techniques, num_transforms)
        
        result = img.copy()
        for transform in transforms:
            try:
                result = transform(result)
            except Exception as e:
                logger.warning(f"Erreur dans transformation {transform.__name__}: {e}")
        
        return result


class DatasetPreparer:
    """Prépare le dataset avec organisation et augmentation"""
    
    def __init__(self, raw_dir="data/raw", output_dir="data/augmented"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.augmentor = DataAugmentor()
        
        self.classes = [
            "piece_identite",
            "releve_notes",
            "facture"
        ]
    
    def organize_raw_data(self):
        """
        Organisation manuelle des données brutes par classe
        Crée la structure et demande à l'utilisateur de placer les fichiers
        """
        organized_dir = Path("data/organized")
        
        logger.info("=== ÉTAPE 1: Organisation des données brutes ===")
        logger.info(f"Création de la structure dans {organized_dir}")
        
        for cls in self.classes:
            (organized_dir / cls).mkdir(parents=True, exist_ok=True)
        
        # Créer un fichier README avec les instructions
        readme_content = """
# Instructions pour organiser les données

Placez vos fichiers (PDF ou images) dans les dossiers correspondants :

- piece_identite/       : CIN recto, CIN verso, cartes d'identité
- releve_bancaire/      : Relevés bancaires
- facture_electricite/  : Factures ONE, RADEM, etc.
- facture_eau/          : Factures d'eau (régies)
- document_employeur/   : Bulletins de paie, attestations employeur, relevés de notes

Vous pouvez mettre des PDF ou des images (JPEG, PNG).
Les PDF seront automatiquement convertis en images.

Minimum recommandé : 5 fichiers par classe
Idéal : 10-20 fichiers par classe

Une fois les fichiers placés, lancez à nouveau ce script.
"""
        
        with open(organized_dir / "README.txt", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        logger.info(f"\n✓ Structure créée dans {organized_dir}")
        logger.info(f"✓ Lisez le fichier {organized_dir}/README.txt pour les instructions")
        logger.info("\n📋 Action requise :")
        logger.info("   1. Placez vos fichiers dans les dossiers appropriés")
        logger.info("   2. Relancez ce script pour l'augmentation\n")
        
        return organized_dir
    
    def check_organized_data(self, organized_dir):
        """Vérifie si les données sont organisées"""
        stats = {}
        
        for cls in self.classes:
            cls_dir = Path(organized_dir) / cls
            if cls_dir.exists():
                files = list(cls_dir.glob("*.*"))
                # Filtrer les fichiers valides
                valid_files = [f for f in files if f.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']]
                stats[cls] = len(valid_files)
            else:
                stats[cls] = 0
        
        return stats
    
    def convert_pdfs_to_images(self, organized_dir):
        """Convertit tous les PDFs en images"""
        logger.info("=== Conversion des PDFs en images ===")
        
        for cls in self.classes:
            cls_dir = Path(organized_dir) / cls
            if not cls_dir.exists():
                continue
            
            pdf_files = list(cls_dir.glob("*.pdf"))
            
            for pdf_file in tqdm(pdf_files, desc=f"Conversion {cls}"):
                try:
                    # Convertir le PDF
                    images = convert_from_path(str(pdf_file), dpi=300)
                    
                    # Sauvegarder chaque page comme image
                    for i, img in enumerate(images):
                        img_path = pdf_file.with_suffix(f'.page{i+1}.png')
                        img.save(img_path, 'PNG')
                    
                    # Supprimer le PDF original
                    pdf_file.unlink()
                    
                except Exception as e:
                    logger.error(f"Erreur conversion {pdf_file}: {e}")
        
        logger.info("✓ Conversion terminée")
    
    def augment_dataset(self, organized_dir, target_per_class=1000):
        """Augmente le dataset pour atteindre target_per_class images par classe"""
        logger.info(f"\n=== ÉTAPE 2: Augmentation des données (cible: {target_per_class}/classe) ===")
        
        # Créer les dossiers de sortie
        for split in ['train', 'val', 'test']:
            for cls in self.classes:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        stats = {}
        
        for cls in self.classes:
            logger.info(f"\nTraitement de la classe: {cls}")
            
            cls_dir = Path(organized_dir) / cls
            if not cls_dir.exists():
                logger.warning(f"  Dossier {cls_dir} non trouvé, passage...")
                continue
            
            # Charger toutes les images originales
            original_images = []
            image_files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        original_images.append(img)
                except Exception as e:
                    logger.error(f"  Erreur lecture {img_file}: {e}")
            
            if len(original_images) == 0:
                logger.warning(f"  Aucune image valide trouvée pour {cls}")
                continue
            
            logger.info(f"  {len(original_images)} images originales chargées")
            
            # Calculer le nombre d'augmentations nécessaires
            num_originals = len(original_images)
            augmentations_per_original = target_per_class // num_originals
            
            logger.info(f"  Génération de {augmentations_per_original} variations par image...")
            
            all_images = []
            
            # Ajouter les images originales
            all_images.extend(original_images)
            
            # Générer les augmentations
            for orig_img in tqdm(original_images, desc=f"  Augmentation {cls}"):
                for _ in range(augmentations_per_original):
                    try:
                        aug_img = self.augmentor.augment(orig_img, num_transforms=random.randint(2, 4))
                        all_images.append(aug_img)
                    except Exception as e:
                        logger.error(f"  Erreur augmentation: {e}")
            
            logger.info(f"  {len(all_images)} images générées au total")
            
            # Mélanger aléatoirement
            random.shuffle(all_images)
            
            # Split train/val/test (70/15/15)
            n_train = int(len(all_images) * 0.70)
            n_val = int(len(all_images) * 0.15)
            
            train_imgs = all_images[:n_train]
            val_imgs = all_images[n_train:n_train+n_val]
            test_imgs = all_images[n_train+n_val:]
            
            # Sauvegarder
            self._save_images(train_imgs, self.output_dir / 'train' / cls, cls)
            self._save_images(val_imgs, self.output_dir / 'val' / cls, cls)
            self._save_images(test_imgs, self.output_dir / 'test' / cls, cls)
            
            stats[cls] = {
                'original': num_originals,
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs),
                'total': len(all_images)
            }
            
            logger.info(f"  ✓ Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        # Sauvegarder les statistiques
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\n✓ Dataset augmenté sauvegardé dans {self.output_dir}")
        return stats
    
    def _save_images(self, images, output_dir, class_name):
        """Sauvegarde une liste d'images"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            output_path = output_dir / f"{class_name}_{i:05d}.png"
            cv2.imwrite(str(output_path), img)
    
    def print_summary(self, stats):
        """Affiche un résumé des statistiques"""
        print("\n" + "="*60)
        print("RÉSUMÉ DU DATASET AUGMENTÉ")
        print("="*60)
        
        for cls, cls_stats in stats.items():
            print(f"\n{cls}:")
            print(f"  Images originales: {cls_stats['original']}")
            print(f"  Train: {cls_stats['train']}")
            print(f"  Val: {cls_stats['val']}")
            print(f"  Test: {cls_stats['test']}")
            print(f"  Total généré: {cls_stats['total']}")
        
        print("\n" + "="*60)


def main():
    """Fonction principale"""
    preparer = DatasetPreparer()
    
    # Étape 1: Organiser les données
    organized_dir = preparer.organize_raw_data()
    
    # Vérifier si des données sont présentes
    stats = preparer.check_organized_data(organized_dir)
    
    total_files = sum(stats.values())
    
    if total_files == 0:
        logger.warning("\n⚠️  Aucun fichier trouvé dans les dossiers organisés")
        logger.info("Veuillez placer vos fichiers dans data/organized/ et relancer le script")
        return
    
    logger.info(f"\n📊 Fichiers trouvés:")
    for cls, count in stats.items():
        logger.info(f"  {cls}: {count} fichiers")
    
    # Demander confirmation
    response = input("\nContinuer avec l'augmentation des données ? (o/n): ")
    if response.lower() != 'o':
        logger.info("Opération annulée")
        return
    
    # Étape 2: Convertir les PDFs
    preparer.convert_pdfs_to_images(organized_dir)
    
    # Étape 3: Augmenter le dataset
    target = int(input("\nNombre d'images cibles par classe (défaut: 1000): ") or "1000")
    aug_stats = preparer.augment_dataset(organized_dir, target_per_class=target)
    
    # Afficher le résumé
    preparer.print_summary(aug_stats)
    
    logger.info("\n✅ Préparation du dataset terminée !")
    logger.info("Vous pouvez maintenant lancer train_full_pipeline.py")


if __name__ == "__main__":
    main()