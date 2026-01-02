# Guide de Démarrage Rapide

## Installation en 5 minutes

### 1. Prérequis
```bash
# Vérifier Python
python --version  # Doit être 3.8+

# Installer Tesseract (Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

### 2. Configuration Environnement
```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

### 3. Initialisation Offline
```bash
# Télécharger les modèles (nécessite internet)
python setup_offline.py
```

### 4. Test Rapide
```bash
# Créer un dossier test
mkdir -p data/raw

# Copier vos PDFs
cp mes_documents/*.pdf data/raw/

# Lancer la classification
python main.py --input data/raw --output outputs
```

## Commandes Utiles

### Classification d'un seul fichier
```bash
python main.py --file document.pdf --output outputs
```

### Classification avec mode verbeux
```bash
python main.py --input data/raw --output outputs --verbose
```

### Vérifier les résultats
```bash
# Voir le rapport
cat outputs/report.txt

# Voir les statistiques
python -c "import json; print(json.dumps(json.load(open('outputs/statistics.json')), indent=2))"
```

## Structure des Résultats

```
outputs/
├── all_results.json     # Détails complets
├── statistics.json      # Stats globales
└── report.txt          # Rapport lisible
```

## Exemples de Résultats

### Document Bien Classé
```json
{
  "predicted_class": "piece_identite",
  "confidence": 0.95,
  "should_review": false,
  "strategy": "perfect_agreement_validated"
}
```

### Document à Revoir
```json
{
  "predicted_class": "facture_eau",
  "confidence": 0.45,
  "should_review": true,
  "strategy": "weighted_fusion"
}
```

## Dépannage Rapide

### Problème: Tesseract non trouvé
```bash
# Vérifier installation
tesseract --version

# Réinstaller si nécessaire
sudo apt-get install --reinstall tesseract-ocr tesseract-ocr-fra
```

### Problème: Modèles non chargés
```bash
# Relancer l'initialisation
python setup_offline.py
```

### Problème: Erreur de mémoire
```bash
# Utiliser un modèle plus léger
# Dans config.yaml, changer:
# backbone: "efficientnet"  # au lieu de "resnet50"
```

## Configuration Rapide

### Changer le seuil de confiance
```yaml
# Dans config.yaml
fusion:
  rejection_threshold: 0.5  # Plus bas = moins de rejets
```

### Désactiver le prétraitement
```yaml
# Dans config.yaml
ocr:
  preprocessing:
    denoise: false
    contrast_enhancement: false
    deskew: false
```

## Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Test d'un module spécifique
pytest tests/test_gabarits.py -v
```

## Benchmarking

```bash
# Inclus dans setup_offline.py
python setup_offline.py

# Voir les temps d'inférence moyens
```

## Performance Attendue

- **Temps par page**: ~2-5 secondes
- **Accuracy cible**: >90% par classe
- **Taux de rejet**: <10%

## Prochaines Étapes

1. Collecter des données d'entraînement
2. Fine-tuner les modèles CV et NLP
3. Ajuster les poids de fusion
4. Optimiser les gabarits
5. Tester sur dataset de validation

## Support

En cas de problème:
1. Consulter README.md complet
2. Vérifier les logs
3. Utiliser mode verbose
4. Consulter le cahier des charges

---

**Bon développement !** 🚀
