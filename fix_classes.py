#!/usr/bin/env python3
"""
Script pour corriger les classes dans tous les fichiers du projet
Remplace les 5 anciennes classes par les 3 nouvelles
"""

import os
from pathlib import Path

# Nouvelles classes depuis config.yaml
NEW_CLASSES = [
    "piece_identite",
    "releve_notes",
    "facture"
]

def fix_cv_model():
    """Corrige cv_model.py"""
    file_path = Path("src/computer_vision/cv_model.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remplacer la liste de classes
    old_classes_block = '''        self.classes = [
            "piece_identite",
            "releve_bancaire",
            "facture_electricite",
            "facture_eau",
            "document_employeur"
        ]'''
    
    new_classes_block = '''        self.classes = [
            "piece_identite",
            "releve_notes",
            "facture"
        ]'''
    
    content = content.replace(old_classes_block, new_classes_block)
    
    # Remplacer aussi num_classes=5 par num_classes=3 dans __init__
    content = content.replace(
        'def __init__(self, num_classes: int = 5,',
        'def __init__(self, num_classes: int = 3,'
    )
    content = content.replace(
        'self.model = HybridCVModel(num_classes=5',
        'self.model = HybridCVModel(num_classes=3'
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {file_path} corrigé")

def fix_nlp_model():
    """Corrige nlp_model.py"""
    file_path = Path("src/nlp/nlp_model.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remplacer la liste de classes
    old_classes_block = '''        self.classes = [
            "piece_identite",
            "releve_bancaire",
            "facture_electricite",
            "facture_eau",
            "document_employeur"
        ]'''
    
    new_classes_block = '''        self.classes = [
            "piece_identite",
            "releve_notes",
            "facture"
        ]'''
    
    content = content.replace(old_classes_block, new_classes_block)
    
    # Remplacer num_classes=5 par num_classes=3
    content = content.replace(
        'def __init__(self, num_classes: int = 5,',
        'def __init__(self, num_classes: int = 3,'
    )
    content = content.replace(
        'self.model = CamembertClassifier(num_classes=5)',
        'self.model = CamembertClassifier(num_classes=3)'
    )
    content = content.replace(
        'model = CamembertClassifier(num_classes=5)',
        'model = CamembertClassifier(num_classes=3)'
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {file_path} corrigé")

def fix_ocr_extractor():
    """Corrige ocr_extractor.py"""
    file_path = Path("src/nlp/ocr_extractor.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remplacer le dictionnaire de keywords
    old_keywords = '''        keywords = {
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
        }'''
    
    new_keywords = '''        keywords = {
            'piece_identite': [
                'identité', 'nationale', 'numéro', 'cnie', 'carte',
                'naissance', 'nationalité', 'marocaine', 'validité',
                'émission', 'prénom', 'nom'
            ],
            'releve_notes': [
                'relevé', 'notes', 'semestre', 'module', 'note',
                'moyenne', 'coefficient', 'crédit', 'ects', 'examen',
                'ensam', 'école', 'étudiant', 'année', 'cp1', 'cp2',
                'validation', 'mention', 'résultat', 'session'
            ],
            'facture': [
                'facture', 'montant', 'total', 'dh', 'dirham',
                'consommation', 'kwh', 'm³', 'électricité', 'eau',
                'one', 'lydec', 'redal', 'radem', 'abonnement',
                'compteur', 'période', 'paiement', 'échéance'
            ]
        }'''
    
    content = content.replace(old_keywords, new_keywords)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {file_path} corrigé")

def main():
    print("=" * 70)
    print("CORRECTION DES CLASSES DANS LE PROJET")
    print("=" * 70)
    print()
    print("Anciennes classes (5):")
    print("  1. piece_identite")
    print("  2. releve_bancaire")
    print("  3. facture_electricite")
    print("  4. facture_eau")
    print("  5. document_employeur")
    print()
    print("Nouvelles classes (3):")
    print("  1. piece_identite")
    print("  2. releve_notes")
    print("  3. facture")
    print()
    print("=" * 70)
    print()
    
    try:
        fix_cv_model()
        fix_nlp_model()
        fix_ocr_extractor()
        
        print()
        print("=" * 70)
        print("✅ CORRECTION TERMINÉE AVEC SUCCÈS")
        print("=" * 70)
        print()
        print("⚠️  IMPORTANT: Vous devez maintenant RÉENTRAÎNER vos modèles!")
        print()
        print("Pourquoi? Vos modèles actuels ont été entraînés avec 5 classes,")
        print("mais nous avons maintenant 3 classes. L'architecture a changé:")
        print("  - Couche de sortie: 5 neurones → 3 neurones")
        print()
        print("Prochaines étapes:")
        print("  1. Réentraîner le modèle CV:")
        print("     python scripts/train_cv.py")
        print()
        print("  2. Réentraîner le modèle NLP:")
        print("     python scripts/train_nlp.py")
        print()
        print("  3. Relancer le pipeline:")
        print("     python main.py --input data/raw --output outputs --verbose")
        print()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
