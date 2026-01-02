#!/usr/bin/env python3
"""
Script principal pour la classification de documents administratifs
Usage:
    python main.py --input data/raw --output outputs
    python main.py --file document.pdf --output outputs
"""
import argparse
import sys
from pathlib import Path
import logging

# Ajouter src au path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import DocumentClassificationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(
        description='Classification automatique de documents administratifs'
    )
    
    # Arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Dossier contenant les PDFs à traiter'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Fichier PDF unique à traiter'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Dossier de sortie pour les résultats (défaut: outputs)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Fichier de configuration (défaut: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbeux'
    )
    
    args = parser.parse_args()
    
    # Vérifier les arguments
    if not args.input and not args.file:
        parser.error("Vous devez spécifier --input ou --file")
    
    if args.input and args.file:
        parser.error("Vous ne pouvez pas spécifier à la fois --input et --file")
    
    # Configurer le niveau de log
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialiser le pipeline
    logger.info("Initialisation du pipeline...")
    try:
        pipeline = DocumentClassificationPipeline(config_path=args.config)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        return 1
    
    # Traiter les documents
    try:
        if args.file:
            # Traitement d'un seul fichier
            logger.info(f"Traitement du fichier: {args.file}")
            results = pipeline.process_single_document(args.file)
            
            # Afficher les résultats
            print("\n" + "="*60)
            print("RÉSULTATS DE CLASSIFICATION")
            print("="*60)
            
            for result in results:
                print(f"\nPage {result['page_number']}:")
                print(f"  Classe prédite: {result['predicted_class']}")
                print(f"  Confiance: {result['confidence']:.3f}")
                print(f"  À revoir: {'Oui' if result['should_review'] else 'Non'}")
                print(f"  Stratégie: {result['strategy']}")
                print(f"  Confiance OCR: {result['ocr_confidence']:.1f}%")
            
            print("\n" + "="*60)
            
        else:
            # Traitement par lots
            logger.info(f"Traitement du dossier: {args.input}")
            batch_results = pipeline.process_batch(args.input, args.output)
            
            # Afficher les statistiques
            stats = batch_results['statistics']
            
            print("\n" + "="*60)
            print("STATISTIQUES DE CLASSIFICATION")
            print("="*60)
            print(f"\nTotal de documents: {stats['total_documents']}")
            print(f"Documents à revoir: {stats['to_review']} ({stats['to_review_percentage']:.1f}%)")
            print(f"Confiance moyenne: {stats['average_confidence']:.3f}")
            
            print("\nDistribution par classe:")
            for cls, count in stats['class_distribution'].items():
                percentage = (count / stats['total_documents']) * 100
                print(f"  {cls}: {count} ({percentage:.1f}%)")
            
            if batch_results['failed_files']:
                print(f"\nFichiers en échec: {len(batch_results['failed_files'])}")
                for failed in batch_results['failed_files']:
                    print(f"  - {failed}")
            
            print(f"\nRésultats sauvegardés dans: {args.output}")
            print("="*60 + "\n")
        
        logger.info("Traitement terminé avec succès")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
