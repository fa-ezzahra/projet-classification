#!/usr/bin/env python3
"""
Script de démonstration interactive du système de classification
Permet de visualiser les features détectées et les scores de chaque module
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.pdf_processor import PDFPreprocessor
from gabarits.detector import GabaritsDetector
from nlp.ocr_extractor import OCRExtractor


def visualize_features(image, features, output_path=None):
    """
    Visualise les features détectées sur l'image
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse des Features Structurelles', fontsize=16, fontweight='bold')
    
    # Image originale
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Image Originale')
    axes[0, 0].axis('off')
    
    # Features textuelles
    h, w = image.shape[:2]
    feature_viz = image.copy()
    
    # Mettre en surbrillance selon la densité de texte
    density = features.get('text_density', 0)
    overlay = np.zeros_like(feature_viz)
    overlay[:, :] = [0, 255, 0]
    cv2.addWeighted(feature_viz, 1, overlay, density * 0.3, 0, feature_viz)
    
    axes[0, 1].imshow(cv2.cvtColor(feature_viz, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Densité de Texte: {density:.2f}')
    axes[0, 1].axis('off')
    
    # Graphique des features
    feature_names = [
        'Photo', 'Table', 'Signature',
        'Text Density', 'Numeric Density'
    ]
    feature_values = [
        features.get('photo_confidence', 0),
        features.get('table_confidence', 0),
        features.get('signature_confidence', 0),
        features.get('text_density', 0),
        features.get('numeric_density', 0)
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = axes[1, 0].barh(feature_names, feature_values, color=colors)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_title('Features Détectées')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, feature_values)):
        axes[1, 0].text(value + 0.02, i, f'{value:.2f}', 
                       va='center', fontweight='bold')
    
    # Informations textuelles
    info_text = f"""
    CARACTÉRISTIQUES DU DOCUMENT
    
    Ratio d'aspect: {features.get('aspect_ratio', 0):.2f}
    
    Photo: {'✓' if features.get('has_photo', 0) > 0.5 else '✗'}
    Table: {'✓' if features.get('has_table', 0) > 0.5 else '✗'}
    Signature: {'✓' if features.get('has_signature', 0) > 0.5 else '✗'}
    
    Densité texte: {features.get('text_density', 0):.2%}
    Densité numérique: {features.get('numeric_density', 0):.2%}
    """
    
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_scores(cv_scores, nlp_scores, gabarit_scores, final_decision, output_path=None):
    """
    Visualise les scores de chaque module et la décision finale
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Scores de Classification Multimodale', fontsize=16, fontweight='bold')
    
    classes = list(cv_scores.keys())
    x = np.arange(len(classes))
    width = 0.25
    
    # Scores Computer Vision
    axes[0, 0].bar(x, [cv_scores[c] for c in classes], width, label='CV', color='#FF6B6B')
    axes[0, 0].set_xlabel('Classes')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Scores Computer Vision')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Scores NLP
    axes[0, 1].bar(x, [nlp_scores[c] for c in classes], width, label='NLP', color='#4ECDC4')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Scores NLP/OCR')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Scores Gabarits
    axes[1, 0].bar(x, [gabarit_scores[c] for c in classes], width, label='Gabarits', color='#45B7D1')
    axes[1, 0].set_xlabel('Classes')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Scores Gabarits')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Décision finale
    final_scores = final_decision['all_scores'].get('fused', {})
    if not final_scores:
        # Si pas de fusion, prendre la moyenne
        final_scores = {c: (cv_scores[c] + nlp_scores[c] + gabarit_scores[c])/3 for c in classes}
    
    bars = axes[1, 1].bar(x, [final_scores[c] for c in classes], width, 
                         label='Final', color='#98D8C8')
    
    # Mettre en évidence la classe prédite
    predicted_idx = classes.index(final_decision['predicted_class'])
    bars[predicted_idx].set_color('#2ECC71')
    bars[predicted_idx].set_edgecolor('black')
    bars[predicted_idx].set_linewidth(2)
    
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title(f"Décision Finale: {final_decision['predicted_class']} "
                        f"(conf={final_decision['confidence']:.2f})")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Ajouter une annotation pour la classe prédite
    axes[1, 1].annotate('PRÉDICTION',
                       xy=(predicted_idx, final_scores[final_decision['predicted_class']]),
                       xytext=(predicted_idx, final_scores[final_decision['predicted_class']] + 0.1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                       fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation des scores sauvegardée: {output_path}")
    else:
        plt.show()
    
    plt.close()


def demo_single_document(pdf_path, output_dir='demo_outputs'):
    """
    Démo complète sur un seul document
    """
    print("="*60)
    print("DÉMONSTRATION DU SYSTÈME DE CLASSIFICATION")
    print("="*60)
    print(f"\nDocument: {pdf_path}")
    
    # Créer le dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialiser les modules
    print("\n[1/5] Initialisation des modules...")
    pdf_processor = PDFPreprocessor(dpi=300)
    gabarit_detector = GabaritsDetector()
    ocr_extractor = OCRExtractor()
    
    # Convertir PDF
    print("\n[2/5] Conversion PDF -> Image...")
    images = pdf_processor.pdf_to_images(pdf_path)
    print(f"  → {len(images)} page(s) détectée(s)")
    
    # Traiter la première page
    image = images[0]
    print(f"  → Dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Prétraiter
    print("\n[3/5] Prétraitement pour OCR...")
    preprocessed = pdf_processor.preprocess_for_ocr(image)
    
    # Extraire texte et features
    print("\n[4/5] Extraction OCR et features...")
    ocr_result = ocr_extractor.process_document(preprocessed)
    print(f"  → Confiance OCR: {ocr_result['ocr_confidence']:.1f}%")
    print(f"  → Mots extraits: {ocr_result['word_count']}")
    
    features = gabarit_detector.extract_all_features(image, ocr_result['corrected_text'])
    print(f"  → {len(features)} features détectées")
    
    # Classification
    print("\n[5/5] Classification...")
    gabarit_scores = gabarit_detector.classify_by_gabarit(image, ocr_result['corrected_text'])
    cv_scores = {k: 0.2 for k in gabarit_scores.keys()}  # Simulation
    nlp_scores = ocr_result['keyword_scores']
    
    # Décision simplifiée pour la démo
    final_decision = {
        'predicted_class': max(gabarit_scores.items(), key=lambda x: x[1])[0],
        'confidence': max(gabarit_scores.values()),
        'strategy': 'demo_mode',
        'should_review': False,
        'all_scores': {
            'cv': cv_scores,
            'nlp': nlp_scores,
            'gabarits': gabarit_scores
        }
    }
    
    # Résultats
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    print(f"\nClasse prédite: {final_decision['predicted_class']}")
    print(f"Confiance: {final_decision['confidence']:.2%}")
    print(f"\nScores par module:")
    print(f"  CV:       {cv_scores[final_decision['predicted_class']]:.2f}")
    print(f"  NLP:      {nlp_scores.get(final_decision['predicted_class'], 0):.2f}")
    print(f"  Gabarits: {gabarit_scores[final_decision['predicted_class']]:.2f}")
    
    # Visualisations
    print("\n[Génération des visualisations...]")
    
    # Sauvegarder l'image originale
    cv2.imwrite(str(output_path / 'original.png'), image)
    
    # Visualiser features
    visualize_features(image, features, output_path / 'features.png')
    
    # Visualiser scores
    visualize_scores(cv_scores, nlp_scores, gabarit_scores, 
                    final_decision, output_path / 'scores.png')
    
    # Sauvegarder les résultats JSON
    with open(output_path / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'ocr_result': {k: v for k, v in ocr_result.items() if k != 'keyword_scores'},
            'features': features,
            'scores': {
                'cv': cv_scores,
                'nlp': nlp_scores,
                'gabarits': gabarit_scores
            },
            'decision': final_decision
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Tous les résultats sauvegardés dans: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Démonstration du système')
    parser.add_argument('pdf_file', help='Fichier PDF à analyser')
    parser.add_argument('--output', '-o', default='demo_outputs', 
                       help='Dossier de sortie')
    
    args = parser.parse_args()
    
    if not Path(args.pdf_file).exists():
        print(f"Erreur: Fichier {args.pdf_file} non trouvé")
        sys.exit(1)
    
    demo_single_document(args.pdf_file, args.output)
