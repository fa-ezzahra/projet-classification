#!/usr/bin/env python3
import sys
from pathlib import Path

print("Test 1: Import pdf2image...")
try:
    from pdf2image import convert_from_path
    print("✓ pdf2image importé")
except ImportError as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

print("\nTest 2: Vérification Poppler...")
try:
    import platform
    if platform.system() == 'Windows':
        print("Système: Windows - Poppler doit être installé manuellement")
        print("Téléchargez: https://github.com/oschwartz10612/poppler-windows/releases/")
    else:
        print(f"Système: {platform.system()}")
except Exception as e:
    print(f"✗ Erreur: {e}")

print("\nTest 3: Conversion d'un PDF...")
pdf_file = input("Entrez le chemin d'un PDF de test (ou Entrée pour skip): ").strip()

if pdf_file and Path(pdf_file).exists():
    try:
        print(f"Tentative de conversion de: {pdf_file}")
        images = convert_from_path(pdf_file, dpi=150, timeout=10)
        print(f"✓ Succès! {len(images)} page(s) converties")
    except Exception as e:
        print(f"✗ Échec: {e}")
        print("\nCause probable: Poppler non installé")
else:
    print("Skip du test de conversion")

print("\n=== Diagnostic terminé ===")