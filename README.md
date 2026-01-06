# Projet de Classification de Documents Administratifs

Système intelligent de classification automatique de documents PDF en 5 catégories principales, fonctionnant **100% offline**.

## 🎯 Objectif

Classifier automatiquement des documents administratifs marocains en utilisant une approche multimodale combinant:
- **Computer Vision** (ResNet50/EfficientNet)
- **NLP** (CamemBERT + OCR Tesseract)
- **Détection de gabarits structurels**
- **Fusion intelligente** des prédictions

## 📋 Classes de Documents

1. **Pièce d'identité** - CNIE (Recto et Verso)
2. **Relevé notes** - Différentes banques marocaines
3. **Facture d'eau et d'électricité** - ONE, RADEM, Lydec, Redal

## 🏗️ Architecture du Système

```
projet_classification/
├── models/                 # Modèles pré-entraînés
│   ├── cv/                # Modèles Computer Vision
│   ├── nlp/               # Modèles NLP (CamemBERT)
│   └── gabarits/          # Définitions des gabarits
├── data/
│   ├── raw/               # Documents PDF bruts
│   ├── organised
            ├── facture
            ├── piece_identite
    │       ├── releve_notes        
    └── augmented/    #pour augmenter data
          ├── facture
          ├── piece_identite
          ├── releve_notes     
      └── small_train/  #pour nlp 
          ├── facture
          ├── piece_identite
          ├── releve_notes   
├── src/
│   ├── preprocessing/     # Prétraitement PDF/images
│   ├── computer_vision/   # Modèles CNN
│   ├── nlp/               # OCR et classification textuelle
│   ├── gabarits/          # Détection de features structurelles
│   ├── fusion/            # Fusion multimodale
│   └── utils/             # Utilitaires et gestionnaire offline
├── tests/                 # Tests unitaires
├── outputs/               # Résultats de classification
├── config.yaml            # Configuration
├── requirements.txt       # Dépendances Python
├── setup_offline.py       # Script d'initialisation
└── main.py                # Point d'entrée principal
└── train_models.py    
└── prepare_dataset.py 
└── configure_tesseract.py  #pour la configuration de tesseract
              
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- Tesseract OCR installé sur le système
- Connexion internet (uniquement pour l'initialisation)

### Installation système de Tesseract

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

#### macOS
```bash
brew install tesseract tesseract-lang
```

#### Windows
Télécharger depuis [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation Python

```bash
# Cloner le projet
cd projet_classification

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Initialiser l'environnement offline (avec connexion internet)
python setup_offline.py
```

Le script `setup_offline.py` va:
1. Vérifier toutes les dépendances
2. Télécharger les modèles pré-entraînés (ResNet50, EfficientNet, CamemBERT)
3. Créer la structure de dossiers
4. Effectuer un benchmarking des modèles

⚠️ **Important**: Cette étape nécessite une connexion internet et peut prendre plusieurs minutes.

## 📖 Utilisation

### Mode Fichier Unique

```bash
python main.py --file chemin/vers/document.pdf --output outputs
```

### Mode Batch (Dossier)

```bash
python main.py --input data/raw --output outputs
```

### Options Avancées

```bash
python main.py --help

Options:
  --input, -i    Dossier contenant les PDFs à traiter
  --file, -f     Fichier PDF unique à traiter
  --output, -o   Dossier de sortie (défaut: outputs)
  --config, -c   Fichier de configuration (défaut: config.yaml)
  --verbose, -v  Mode verbeux
```

### Exemple Complet

```bash
# Placer vos PDFs dans data/raw/
cp mes_documents/*.pdf data/raw/

# Lancer la classification
python main.py --input data/raw --output outputs --verbose

# Les résultats seront dans outputs/
# - all_results.json : Tous les résultats détaillés
# - statistics.json : Statistiques globales
# - report.txt : Rapport textuel
```

## 📊 Résultats

Le système génère trois types de fichiers de sortie:

### 1. all_results.json
Contient tous les détails de classification pour chaque page:
```json
{
  "pdf_path": "data/raw/document.pdf",
  "page_number": 1,
  "predicted_class": "piece_identite",
  "confidence": 0.95,
  "should_review": false,
  "strategy": "perfect_agreement_validated",
  "ocr_confidence": 87.5,
  "gabarit_features": {...},
  "all_scores": {...}
}
```

### 2. statistics.json
Statistiques globales sur le batch traité:
```json
{
  "total_documents": 100,
  "to_review": 5,
  "to_review_percentage": 5.0,
  "average_confidence": 0.89,
  "class_distribution": {...},
  "strategies_used": {...}
}
```

### 3. report.txt
Rapport textuel lisible

## ⚙️ Configuration

Le fichier `config.yaml` permet de configurer tous les aspects du système:

```yaml
# Modèles Computer Vision
computer_vision:
  backbone: "resnet50"  # ou "efficientnet"
  input_size: [224, 224]
  batch_size: 16

# Modèles NLP
nlp:
  model_name: "camembert-base"
  max_length: 512

# Paramètres OCR
ocr:
  language: "fra"
  dpi: 300
  preprocessing:
    denoise: true
    contrast_enhancement: true
    deskew: true

# Paramètres de fusion
fusion:
  confidence_thresholds:
    high: 0.9
    medium: 0.7
    low: 0.5
  weights:
    cv: 0.4
    nlp: 0.3
    gabarits: 0.3
```

## 🔧 Modules Techniques

### Module 1: Prétraitement (preprocessing/)
- Conversion PDF → Images (pdf2image)
- Débruitage (OpenCV)
- Amélioration du contraste (CLAHE)
- Correction d'inclinaison (Hough Transform)

### Module 2: Gabarits (gabarits/)
- Détection de zones photo (Haar Cascades)
- Analyse de structure tabulaire (Hough Lines)
- Calcul de densité de texte
- Détection de zones de signature
- Features: aspect_ratio, has_photo, has_table, text_density, etc.

### Module 3: Computer Vision (computer_vision/)
- Backbone: ResNet50 ou EfficientNet
- Extraction de features visuelles haute dimension
- Classification basée sur l'apparence globale

### Module 4: NLP (nlp/)
- OCR: Tesseract avec langue française
- Extraction de mots-clés par catégorie
- Classification basée sur le contenu sémantique
- Post-correction des erreurs OCR courantes

### Module 5: Fusion Multimodale (fusion/)
Stratégies de décision:
1. **Accord parfait**: CV et NLP concordent avec haute confiance
2. **CV fort + Validation gabarits**: CV confiant validé par les gabarits
3. **NLP fort + Motifs textuels**: NLP confiant avec mots-clés spécifiques
4. **Fusion pondérée**: Combinaison pondérée de tous les scores
5. **Règles métier**: Validation finale par contraintes métier

## 📈 Métriques d'Évaluation

Le système calcule automatiquement:
- **Accuracy globale** et par classe
- **Precision, Recall, F1-score** par classe
- **Taux de rejet** (documents à revoir manuellement)
- **Confiance moyenne** des prédictions
- **Distribution des stratégies** utilisées

Objectif: **90%+ d'accuracy par classe**

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_preprocessing.py
pytest tests/test_gabarits.py
pytest tests/test_fusion.py
```

## 📝 Exemples de Gabarits

### Pièce d'Identité
- Ratio d'aspect: 1.5 - 1.7 (format carte)
- Présence de photo obligatoire
- Densité de texte moyenne (0.3 - 0.5)

### Relevé Bancaire
- Structure tabulaire forte
- Lignes verticales pour les colonnes
- Haute densité numérique (montants)

### Factures
- Structure tabulaire
- Unités de mesure spécifiques (kWh, m³)
- Périodes de facturation

### Document Employeur
- Zone de signature probable
- Mentions salariales
- Format A4 standard

## 🔍 Debugging

Le système inclut plusieurs outils de debugging:

```python
# Visualiser les features détectées
from src.gabarits.detector import GabaritsDetector
detector = GabaritsDetector()
features = detector.extract_all_features(image, text)
print(features)

# Vérifier les scores de chaque module
cv_scores = {...}
nlp_scores = {...}
gabarit_scores = {...}
# Tous inclus dans all_results.json
```

## 🚧 Limitations Connues

1. **Documents manuscrits**: L'OCR peut échouer sur l'écriture manuscrite
2. **Qualité image**: Les documents très dégradés nécessitent une révision manuelle
3. **Formats non standards**: Les documents avec mise en page inhabituelle peuvent être mal classés
4. **Multi-pages complexes**: Les PDFs mélangeant plusieurs types de documents

## 🛠️ Optimisations

### Performance
- Cache des modèles chargés
- Traitement par batch
- Multithreading pour l'OCR

### Mémoire
- Chargement lazy des modèles
- Libération de la mémoire entre les batches
- Options de modèles légers (EfficientNet-B0)

## 📚 Documentation Complète

Pour plus de détails, consultez:
- [Guide d'architecture](docs/architecture.md)
- [Guide des gabarits](docs/gabarits.md)
- [API Reference](docs/api.md)

## 👥 Contribution

Ce projet est réalisé dans le cadre du module CV/NLP à ENSAM Rabat.

### Équipe
- Définir les rôles dans votre trinôme
- Utiliser Git pour la collaboration
- Synchronisation régulière

## 📄 Licence

Projet académique - ENSAM Rabat 2025

## 🆘 Support

En cas de problème:
1. Vérifier que Tesseract est bien installé: `tesseract --version`
2. Vérifier les dépendances: `python setup_offline.py`
3. Consulter les logs dans `logs/`
4. Utiliser le mode verbose: `--verbose`

## 🎓 Ressources Pédagogiques

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

**Version**: 1.0.0  
**Date**: Janvier 2026  
**Auteurs**: [Votre Trinôme ENSAM]
