#!/usr/bin/env python3
"""
Script de conversion des PDFs de releve_notes en images PNG
"""
from pdf2image import convert_from_path
from pathlib import Path
import sys

def convert_pdfs():
    """Convertit tous les PDFs de releve_notes en images PNG"""
    
    pdf_dir = Path('data/organized/releve_notes')
    
    if not pdf_dir.exists():
        print(f"❌ Dossier {pdf_dir} non trouvé!")
        return False
    
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    if not pdf_files:
        print(f"❌ Aucun PDF trouvé dans {pdf_dir}")
        return False
    
    print(f"📄 {len(pdf_files)} PDFs trouvés")
    print("="*60)
    
    success_count = 0
    error_count = 0
    
    for pdf_file in pdf_files:
        print(f"\n🔄 Conversion de: {pdf_file.name}")
        try:
            # Convertir le PDF (dpi=300 pour bonne qualité)
            # Spécifier le chemin de poppler si nécessaire
            poppler_path = None
            
            # Chemins possibles de poppler sur Windows
            possible_paths = [
                r"C:\Users\hp\Documents\S5\projet-classification\poppler\Library\bin",
                r"C:\Users\hp\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin",
                r"C:\Program Files\poppler\Library\bin",
                r"C:\poppler\Library\bin",
                r"C:\Program Files\poppler-23.11.0\Library\bin",
                r"C:\Program Files (x86)\poppler\Library\bin",
            ]
            
            for path in possible_paths:
                from pathlib import Path as P
                if P(path).exists():
                    poppler_path = path
                    print(f"   ℹ️  Poppler trouvé: {path}")
                    break
            
            images = convert_from_path(str(pdf_file), dpi=300, poppler_path=poppler_path)
            
            print(f"   ✓ {len(images)} page(s) détectée(s)")
            
            # Sauvegarder chaque page
            for i, img in enumerate(images):
                # Nom: ELAZZOUZI_1ere_CI.page1.png
                img_path = pdf_file.with_suffix(f'.page{i+1}.png')
                img.save(img_path, 'PNG')
                print(f"   ✓ Sauvegardé: {img_path.name}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ ERREUR: {e}")
            error_count += 1
    
    print("\n" + "="*60)
    print(f"✅ Succès: {success_count}/{len(pdf_files)}")
    if error_count > 0:
        print(f"❌ Erreurs: {error_count}/{len(pdf_files)}")
    
    # Vérifier les images créées
    png_files = list(pdf_dir.glob('*.png'))
    print(f"\n📊 Total d'images PNG maintenant: {len(png_files)}")
    
    if png_files:
        print("\n✅ Conversion réussie!")
        print("\nProchaine étape:")
        print("  python prepare_dataset.py")
        return True
    else:
        print("\n⚠️ Aucune image n'a été créée")
        return False


if __name__ == "__main__":
    try:
        success = convert_pdfs()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print("\n❌ ERREUR D'IMPORT:")
        print(str(e))
        print("\n📦 Installez les dépendances manquantes:")
        print("   pip install pdf2image")
        print("\n⚠️ Vous devez aussi installer poppler:")
        print("   - Windows: téléchargez depuis https://github.com/oschwaldp/poppler-windows/releases/")
        print("   - Ou avec chocolatey: choco install poppler")
        sys.exit(1)