# PROJET DE CLASSIFICATION DE DOCUMENTS ADMINISTRATIFS
## Système Complet Offline - ENSAM Rabat 2025

---

## 📦 CONTENU DU PROJET

Le projet complet contient tous les fichiers nécessaires pour démarrer immédiatement :

### Structure des Fichiers

```
projet_classification/
│
├── 📄 README.md                    # Documentation complète
├── 📄 QUICKSTART.md                # Guide de démarrage rapide
├── 📄 requirements.txt             # Dépendances Python
├── 📄 config.yaml                  # Configuration du système
├── 📄 .gitignore                   # Fichiers à ignorer dans Git
│
├── 🔧 setup_offline.py             # Script d'initialisation (à exécuter en premier)
├── 🚀 main.py                      # Point d'entrée principal
├── 🎨 demo.py                      # Script de démonstration avec visualisations
│
├── 📁 src/                         # Code source
│   ├── preprocessing/              # Prétraitement PDF et images
│   │   └── pdf_processor.py       # Conversion PDF, débruitage, contraste
│   ├── computer_vision/            # Modèles Computer Vision
│   ├── nlp/                        # NLP et OCR
│   │   └── ocr_extractor.py       # Extraction texte + classification
│   ├── gabarits/                   # Système de gabarits
│   │   └── detector.py            # Détection features structurelles
│   ├── fusion/                     # Fusion multimodale
│   │   └── multimodal_fusion.py   # Stratégies de décision
│   ├── utils/                      # Utilitaires
│   │   └── offline_manager.py     # Gestionnaire modèles offline
│   └── pipeline.py                 # Pipeline complet
│
├── 📁 tests/                       # Tests unitaires
│   └── test_gabarits.py           # Tests du module gabarits
│
├── 📁 models/                      # Modèles pré-entraînés (à télécharger)
│   ├── cv/                         # ResNet50, EfficientNet
│   ├── nlp/                        # CamemBERT
│   └── gabarits/                   # Définitions des gabarits
│
├── 📁 data/                        # Données
│   ├── raw/                        # PDFs bruts à classifier
│   ├── processed/                  # Images prétraitées
│   └── annotations/                # Labels et métadonnées
│
└── 📁 outputs/                     # Résultats de classification
    ├── all_results.json            # Détails complets
    ├── statistics.json             # Statistiques globales
    └── report.txt                  # Rapport textuel
```

---

## 🎯 FONCTIONNALITÉS IMPLÉMENTÉES

### ✅ Module 1 : Configuration Offline
- ✓ Gestionnaire de modèles avec cache intelligent
- ✓ Chargement offline de ResNet50, EfficientNet, CamemBERT
- ✓ Vérification automatique des dépendances
- ✓ Système de repli avec modèles légers

### ✅ Module 2 : Système de Gabarits
- ✓ Détection de zones photo (Haar Cascades)
- ✓ Analyse de structure tabulaire (Hough Transform)
- ✓ Calcul de densité de texte
- ✓ Détection de zones de signature
- ✓ Calcul du ratio d'aspect
- ✓ Scores de correspondance par gabarit

### ✅ Module 3 : Prétraitement
- ✓ Conversion PDF → Images (pdf2image)
- ✓ Débruitage (Non-Local Means)
- ✓ Amélioration contraste (CLAHE)
- ✓ Correction d'inclinaison (Hough Lines)
- ✓ Binarisation adaptative

### ✅ Module 4 : NLP/OCR
- ✓ Extraction texte (Tesseract OCR)
- ✓ Prétraitement pour OCR
- ✓ Correction post-OCR
- ✓ Extraction mots-clés par catégorie
- ✓ Classification basée sur mots-clés
- ✓ Extraction d'informations structurées

### ✅ Module 5 : Fusion Multimodale
- ✓ Stratégie 1 : Accord parfait CV + NLP
- ✓ Stratégie 2 : CV fort + Validation gabarits
- ✓ Stratégie 3 : NLP fort + Motifs textuels
- ✓ Stratégie 4 : Fusion pondérée
- ✓ Règles métier par type de document
- ✓ Système de confiance et de rejet

### ✅ Module 6 : Pipeline Principal
- ✓ Traitement fichier unique
- ✓ Traitement par batch
- ✓ Gestion d'erreurs robuste
- ✓ Génération de rapports
- ✓ Interface CLI complète

---

## 🚀 GUIDE D'UTILISATION

### Installation Rapide

```bash
# 1. Installer Tesseract (Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# 2. Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Initialiser (nécessite internet, une seule fois)
python setup_offline.py
```

### Utilisation

```bash
# Classifier un seul fichier
python main.py --file document.pdf --output outputs

# Classifier un dossier
python main.py --input data/raw --output outputs

# Mode verbeux
python main.py --input data/raw --output outputs --verbose

# Démonstration avec visualisations
python demo.py document.pdf --output demo_outputs
```

---

## 📊 CLASSES DE DOCUMENTS SUPPORTÉES

1. **Pièce d'Identité** (CNIE Recto/Verso)
   - Features: Photo + Format carte + Densité texte moyenne
   
2. **Relevé Bancaire**
   - Features: Structure tabulaire + Montants financiers
   
3. **Facture d'Électricité** (ONE, RADEM, Lydec, Redal)
   - Features: Table + Unités kWh + Périodes
   
4. **Facture d'Eau**
   - Features: Table + Unités m³ + Consommation
   
5. **Document Employeur** (Bulletins, Attestations)
   - Features: Signature + Mentions salariales

---

## 🎨 POINTS FORTS DU SYSTÈME

### 1. **Approche Multimodale Robuste**
- Combine 3 sources d'information : CV, NLP, Gabarits
- Validation croisée entre les modules
- Règles métier spécifiques par type

### 2. **Système de Confiance Intelligent**
- Scores de confiance par prédiction
- Rejet automatique des cas ambigus
- Stratégies de décision adaptatives

### 3. **100% Offline**
- Tous les modèles chargés localement
- Pas de dépendance internet après setup
- Cache intelligent des modèles

### 4. **Robustesse**
- Prétraitement avancé des images
- Correction post-OCR
- Gestion d'erreurs complète
- Timeouts et reprise sur erreur

### 5. **Extensibilité**
- Architecture modulaire
- Configuration YAML flexible
- Facile d'ajouter de nouvelles classes
- System de plugins pour gabarits

---

## 📈 MÉTRIQUES ET PERFORMANCES

### Objectifs de Performance
- **Accuracy**: >90% par classe
- **Temps par page**: 2-5 secondes
- **Taux de rejet**: <10%

### Métriques Calculées
- Accuracy globale et par classe
- Precision, Recall, F1-score
- Confiance moyenne
- Distribution des stratégies
- Taux de documents à revoir

---

## 🧪 TESTS

```bash
# Tous les tests
pytest tests/ -v

# Test spécifique
pytest tests/test_gabarits.py -v

# Couverture de code
pytest --cov=src tests/
```

---

## 🔧 PERSONNALISATION

### Modifier les Seuils

```yaml
# Dans config.yaml
fusion:
  confidence_thresholds:
    high: 0.9      # Confiance haute
    medium: 0.7    # Confiance moyenne
    low: 0.5       # Confiance basse
  rejection_threshold: 0.5  # Seuil de rejet
```

### Ajuster les Poids

```yaml
fusion:
  weights:
    cv: 0.4          # Poids Computer Vision
    nlp: 0.3         # Poids NLP
    gabarits: 0.3    # Poids Gabarits
```

### Changer le Modèle

```yaml
computer_vision:
  backbone: "efficientnet"  # ou "resnet50"
```

---

## 📚 RESSOURCES PÉDAGOGIQUES

### Technologies Utilisées
- **PyTorch** : Modèles deep learning
- **Transformers** : CamemBERT
- **OpenCV** : Traitement d'images
- **Tesseract** : OCR
- **pdf2image** : Conversion PDF

### Concepts Clés
- Réseaux de neurones convolutionnels (CNN)
- Transformers pour NLP
- Feature engineering
- Fusion multimodale
- Règles métier

---

## 🎓 POUR ALLER PLUS LOIN

### Améliorations Possibles

1. **Fine-tuning des Modèles**
   - Collecter dataset annoté
   - Fine-tuner ResNet50 sur vos données
   - Fine-tuner CamemBERT sur textes administratifs

2. **Optimisation des Gabarits**
   - Ajouter plus de features spécifiques
   - Apprendre les gabarits automatiquement
   - Utiliser des embeddings visuels

3. **Interface Utilisateur**
   - Ajouter interface web (Streamlit/Flask)
   - Système de validation manuelle
   - Dashboard de monitoring

4. **Performance**
   - Quantization des modèles
   - Batch processing optimisé
   - GPU acceleration

---

## 🤝 TRAVAIL EN ÉQUIPE

### Organisation Recommandée

1. **Chef d'équipe** : Coordination, intégration
2. **Membre 1** : CV + Prétraitement
3. **Membre 2** : NLP + OCR
4. **Membre 3** : Gabarits + Fusion

### Workflow Git

```bash
# Créer une branche
git checkout -b feature/mon-module

# Développer et commiter
git add .
git commit -m "Implémentation module X"

# Pousser et créer PR
git push origin feature/mon-module
```

---

## 📞 SUPPORT

### En cas de problème

1. Consulter `README.md` et `QUICKSTART.md`
2. Vérifier les logs dans `logs/`
3. Utiliser mode verbose : `--verbose`
4. Lancer les tests : `pytest tests/ -v`
5. Vérifier Tesseract : `tesseract --version`

---

## ✨ CONCLUSION

Ce projet fournit une **base solide et fonctionnelle** pour la classification de documents administratifs. Tous les modules essentiels sont implémentés et testés.

### Prochaines Étapes Recommandées

1. ✅ Installer et tester le système
2. ✅ Collecter des données d'entraînement
3. ✅ Fine-tuner les modèles
4. ✅ Optimiser les gabarits
5. ✅ Évaluer sur dataset de validation
6. ✅ Ajuster les paramètres
7. ✅ Déployer en production

**Bon développement !** 🚀

---

**Projet réalisé dans le cadre du module CV/NLP**  
**ENSAM Rabat - Janvier 2026**  
**Par : [Votre Trinôme]**
