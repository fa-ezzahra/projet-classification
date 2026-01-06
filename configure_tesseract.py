#!/usr/bin/env python3
"""
Script de détection et configuration de Tesseract
"""
import subprocess
import sys
from pathlib import Path

def find_tesseract():
    """Trouve l'emplacement de tesseract.exe"""
    
    print("🔍 Recherche de Tesseract...")
    
    # Chemins courants sur Windows
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\tesseract\tesseract.exe",
        r"C:\Users\hp\Documents\S5\projet-classification\tesseract\tesseract.exe",
    ]
    
    # Vérifier les chemins courants
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Tesseract trouvé: {path}")
            return path
    
    # Essayer avec 'where' (Windows)
    try:
        result = subprocess.run(['where', 'tesseract'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            path = result.stdout.strip().split('\n')[0]
            print(f"✅ Tesseract trouvé via 'where': {path}")
            return path
    except:
        pass
    
    print("❌ Tesseract introuvable automatiquement")
    return None


def test_tesseract(tesseract_path):
    """Teste si Tesseract fonctionne"""
    try:
        result = subprocess.run([tesseract_path, '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Tesseract fonctionne: {version}")
            return True
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
    
    return False


def configure_pytesseract(tesseract_path):
    """Configure pytesseract avec le chemin trouvé"""
    
    ocr_file = Path("src/nlp/ocr_extractor.py")
    
    if not ocr_file.exists():
        print(f"❌ Fichier {ocr_file} introuvable")
        return False
    
    print(f"\n📝 Configuration de {ocr_file}...")
    
    # Lire le fichier
    with open(ocr_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier si déjà configuré
    if 'pytesseract.pytesseract.tesseract_cmd' in content:
        print("⚠️  Configuration déjà présente")
        
        # Mettre à jour le chemin
        import re
        pattern = r"pytesseract\.pytesseract\.tesseract_cmd\s*=\s*r?['\"].*?['\"]"
        new_line = f'pytesseract.pytesseract.tesseract_cmd = r"{tesseract_path}"'
        
        if re.search(pattern, content):
            content = re.sub(pattern, new_line, content)
            print(f"✅ Chemin mis à jour vers: {tesseract_path}")
        else:
            print("❌ Impossible de trouver la ligne à mettre à jour")
            return False
    else:
        # Ajouter la configuration après l'import
        import_line = "import pytesseract"
        if import_line in content:
            config_line = f'\n# Configuration du chemin Tesseract\npytesseract.pytesseract.tesseract_cmd = r"{tesseract_path}"\n'
            content = content.replace(import_line, import_line + config_line)
            print("✅ Configuration ajoutée")
        else:
            print("❌ Impossible de trouver la ligne d'import pytesseract")
            return False
    
    # Sauvegarder
    with open(ocr_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fichier {ocr_file} mis à jour")
    return True


def main():
    print("="*60)
    print("CONFIGURATION DE TESSERACT POUR LE PROJET")
    print("="*60)
    
    # Trouver Tesseract
    tesseract_path = find_tesseract()
    
    if not tesseract_path:
        print("\n❌ TESSERACT INTROUVABLE")
        print("\nOptions:")
        print("1. Installez Tesseract depuis: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Ou indiquez manuellement le chemin")
        
        manual_path = input("\nEntrez le chemin de tesseract.exe (ou laissez vide pour annuler): ").strip()
        
        if manual_path and Path(manual_path).exists():
            tesseract_path = manual_path
        else:
            print("❌ Opération annulée")
            return 1
    
    # Tester Tesseract
    print(f"\n🧪 Test de Tesseract...")
    if not test_tesseract(tesseract_path):
        print("❌ Tesseract ne fonctionne pas correctement")
        return 1
    
    # Configurer pytesseract
    if configure_pytesseract(tesseract_path):
        print("\n" + "="*60)
        print("✅ CONFIGURATION TERMINÉE")
        print("="*60)
        print("\n📋 Prochaines étapes:")
        print("  1. Testez l'OCR: python -c \"from src.nlp.ocr_extractor import OCRExtractor; print('OCR OK')\"")
        print("  2. Relancez l'entraînement NLP ou le pipeline principal")
        return 0
    else:
        print("\n❌ Échec de la configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())
