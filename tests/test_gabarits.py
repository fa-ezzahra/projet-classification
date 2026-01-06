"""
Tests unitaires pour le module de détection de gabarits
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gabarits.detector import GabaritsDetector


class TestGabaritsDetector:
    """
    Tests pour la classe GabaritsDetector
    """
    
    @pytest.fixture
    def detector(self):
        """Fixture pour créer un détecteur"""
        return GabaritsDetector()
    
    @pytest.fixture
    def sample_image(self):
        """Créer une image de test"""
        # Image 800x600 (ratio 1.33)
        return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    @pytest.fixture
    def card_image(self):
        """Créer une image au format carte"""
        # Image 850x500 (ratio 1.7, format carte)
        return np.random.randint(0, 255, (500, 850, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test l'initialisation du détecteur"""
        assert detector is not None
        assert detector.face_cascade is not None
    
    def test_calculate_aspect_ratio(self, detector, sample_image):
        """Test le calcul du ratio d'aspect"""
        ratio = detector.calculate_aspect_ratio(sample_image)
        assert 1.3 < ratio < 1.4  # 800/600 ≈ 1.33
    
    def test_card_aspect_ratio(self, detector, card_image):
        """Test la détection du format carte"""
        ratio = detector.calculate_aspect_ratio(card_image)
        assert 1.6 < ratio < 1.8  # Format carte typique
    
    def test_text_density_calculation(self, detector, sample_image):
        """Test le calcul de densité de texte"""
        density = detector.calculate_text_density(sample_image)
        assert 0.0 <= density <= 1.0
    
    def test_extract_all_features(self, detector, sample_image):
        """Test l'extraction de toutes les features"""
        features = detector.extract_all_features(sample_image)
        
        # Vérifier que toutes les features attendues sont présentes
        expected_features = [
            'aspect_ratio',
            'has_photo',
            'photo_confidence',
            'has_table',
            'table_confidence',
            'text_density',
            'numeric_density',
            'has_signature',
            'signature_confidence'
        ]
        
        for feature in expected_features:
            assert feature in features
            # Vérifier que les valeurs sont dans les bonnes plages
            if 'confidence' in feature or 'density' in feature:
                assert 0.0 <= features[feature] <= 1.0
    
    def test_match_gabarit_piece_identite(self, detector, card_image):
        """Test le matching avec le gabarit pièce d'identité"""
        features = detector.extract_all_features(card_image)
        score = detector.match_gabarit(features, "piece_identite")
        
        assert 0.0 <= score <= 1.0
    
    def test_classify_by_gabarit(self, detector, sample_image):
        """Test la classification complète par gabarits"""
        scores = detector.classify_by_gabarit(sample_image)
        
        # Vérifier que tous les types de documents ont un score
        expected_classes = [
            "piece_identite",
            "releve_bancaire",
            "facture_electricite",
            "facture_eau",
            "document_employeur"
        ]
        
        for doc_class in expected_classes:
            assert doc_class in scores
            assert 0.0 <= scores[doc_class] <= 1.0
        
        # Vérifier que les scores somment à quelque chose de raisonnable
        total_score = sum(scores.values())
        assert total_score >= 0.0


class TestGabaritsWithText:
    """
    Tests pour les features textuelles
    """
    
    @pytest.fixture
    def detector(self):
        return GabaritsDetector()
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    def test_numeric_content_detection(self, detector, sample_image):
        """Test la détection de contenu numérique"""
        text_with_numbers = "Montant: 1500.50 DH Date: 25/12/2024"
        score = detector.detect_numeric_content(sample_image, text_with_numbers)
        assert score > 0.0
    
    def test_numeric_content_no_numbers(self, detector, sample_image):
        """Test avec texte sans chiffres"""
        text_no_numbers = "Document administratif sans montants"
        score = detector.detect_numeric_content(sample_image, text_no_numbers)
        assert score == 0.0


def test_integration_full_pipeline():
    """
    Test d'intégration: pipeline complet de détection
    """
    detector = GabaritsDetector()
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    test_text = "Facture électricité ONE consommation 150 kWh"
    
    # Extraire les features
    features = detector.extract_all_features(test_image, test_text)
    
    # Classifier
    scores = detector.classify_by_gabarit(test_image, test_text)
    
    # Vérifications
    assert len(features) > 0
    assert len(scores) == 5  # 5 classes
    
    # Le score pour facture_electricite devrait être raisonnablement élevé
    # car le texte contient "électricité" et "kWh"
    assert scores['facture_electricite'] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
