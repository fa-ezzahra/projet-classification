"""
Module Computer Vision Hybride
Combine CNN (ResNet50) avec features de gabarits pour classification
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridCVModel(nn.Module):
    """
    Modèle hybride combinant ResNet50 et features de gabarits
    """
    
    def __init__(self, num_classes: int = 3, gabarit_features_dim: int = 9):
        """
        Initialise le modèle hybride
        
        Args:
            num_classes: Nombre de classes de documents
            gabarit_features_dim: Dimension des features de gabarits
        """
        super(HybridCVModel, self).__init__()
        
        # Branch 1: ResNet50 backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Supprimer la dernière couche fully connected
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        resnet_output_dim = 2048  # ResNet50 output
        
        # Branch 2: Gabarits features processing
        self.gabarit_fc = nn.Sequential(
            nn.Linear(gabarit_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        fusion_dim = resnet_output_dim + 64  # 2048 + 64 = 2112
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"HybridCVModel initialisé: {num_classes} classes")
    
    def forward(self, image: torch.Tensor, gabarit_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image: Tensor d'images [batch_size, 3, 224, 224]
            gabarit_features: Tensor de features [batch_size, gabarit_features_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Branch ResNet
        resnet_out = self.resnet_features(image)
        resnet_out = torch.flatten(resnet_out, 1)  # [batch_size, 2048]
        
        # Branch Gabarits
        gabarit_out = self.gabarit_fc(gabarit_features)  # [batch_size, 64]
        
        # Fusion
        fused = torch.cat([resnet_out, gabarit_out], dim=1)  # [batch_size, 2112]
        
        # Classification
        logits = self.fusion_classifier(fused)  # [batch_size, num_classes]
        
        return logits


class CVClassifier:
    """
    Classificateur Computer Vision avec preprocessing
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialise le classificateur
        
        Args:
            model_path: Chemin vers le modèle fine-tuné (optionnel)
            device: Device PyTorch (cpu ou cuda)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialiser le modèle
        self.model = HybridCVModel(num_classes=3, gabarit_features_dim=9)
        
        # Charger les poids fine-tunés si disponibles
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Modèle chargé depuis {model_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle: {e}")
                logger.warning("Utilisation du modèle pré-entraîné sans fine-tuning")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transformations d'images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Classes
        self.classes = [
            "piece_identite",
            "releve_notes",
            "facture"
        ]
        
        logger.info(f"CVClassifier initialisé sur {self.device}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraite une image pour le modèle
        
        Args:
            image: Image numpy array [H, W, C]
            
        Returns:
            Tensor preprocessé [1, 3, 224, 224]
        """
        # Appliquer les transformations
        img_tensor = self.transform(image)
        
        # Ajouter batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def preprocess_gabarit_features(self, features: Dict[str, float]) -> torch.Tensor:
        """
        Prétraite les features de gabarits
        
        Args:
            features: Dictionnaire de features
            
        Returns:
            Tensor de features [1, 9]
        """
        # Ordre des features
        feature_order = [
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
        
        # Extraire dans le bon ordre avec valeurs par défaut
        feature_values = [features.get(f, 0.0) for f in feature_order]
        
        # Normaliser aspect_ratio (généralement entre 0.5 et 2.0)
        feature_values[0] = min(max(feature_values[0], 0.5), 2.0) / 2.0
        
        # Convertir en tensor
        features_tensor = torch.FloatTensor(feature_values).unsqueeze(0)
        
        return features_tensor.to(self.device)
    
    def predict(self, image: np.ndarray, gabarit_features: Dict[str, float]) -> Dict[str, float]:
        """
        Prédit la classe d'un document
        
        Args:
            image: Image numpy array
            gabarit_features: Features de gabarits
            
        Returns:
            Dictionnaire de scores par classe
        """
        with torch.no_grad():
            # Prétraiter
            img_tensor = self.preprocess_image(image)
            features_tensor = self.preprocess_gabarit_features(gabarit_features)
            
            # Forward pass
            logits = self.model(img_tensor, features_tensor)
            
            # Softmax pour obtenir les probabilités
            probs = torch.softmax(logits, dim=1)
            
            # Convertir en dictionnaire
            scores = {}
            for i, class_name in enumerate(self.classes):
                scores[class_name] = probs[0, i].item()
        
        return scores
    
    def predict_batch(self, images: list, gabarit_features_list: list) -> list:
        """
        Prédit un batch d'images
        
        Args:
            images: Liste d'images numpy
            gabarit_features_list: Liste de dictionnaires de features
            
        Returns:
            Liste de dictionnaires de scores
        """
        results = []
        
        # Traiter par batch de 16
        batch_size = 16
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_features = gabarit_features_list[i:i+batch_size]
            
            # Prétraiter le batch
            img_tensors = torch.cat([
                self.preprocess_image(img) for img in batch_images
            ], dim=0)
            
            features_tensors = torch.cat([
                self.preprocess_gabarit_features(feats) for feats in batch_features
            ], dim=0)
            
            with torch.no_grad():
                # Forward pass
                logits = self.model(img_tensors, features_tensors)
                probs = torch.softmax(logits, dim=1)
                
                # Convertir en liste de dictionnaires
                for j in range(probs.shape[0]):
                    scores = {}
                    for k, class_name in enumerate(self.classes):
                        scores[class_name] = probs[j, k].item()
                    results.append(scores)
        
        return results


class CVModelTrainer:
    """
    Entraîneur pour le modèle CV hybride
    """
    
    def __init__(self, model: HybridCVModel, device: str = None):
        """
        Initialise l'entraîneur
        
        Args:
            model: Modèle à entraîner
            device: Device PyTorch
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss et optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        logger.info("CVModelTrainer initialisé")
    
    def train_epoch(self, train_loader) -> float:
        """
        Entraîne une époque
        
        Args:
            train_loader: DataLoader d'entraînement
            
        Returns:
            Loss moyenne
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, gabarit_features, labels) in enumerate(train_loader):
            # Déplacer sur device
            images = images.to(self.device)
            gabarit_features = gabarit_features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images, gabarit_features)
            
            # Calculer loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Valide le modèle
        
        Args:
            val_loader: DataLoader de validation
            
        Returns:
            Tuple (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, gabarit_features, labels in val_loader:
                # Déplacer sur device
                images = images.to(self.device)
                gabarit_features = gabarit_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images, gabarit_features)
                
                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 50, 
              save_path: str = "models/cv/hybrid_model.pth"):
        """
        Boucle d'entraînement complète
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            epochs: Nombre d'époques
            save_path: Chemin pour sauvegarder le meilleur modèle
        """
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entraîner
            train_loss = self.train_epoch(train_loader)
            
            # Valider
            val_loss, val_acc = self.validate(val_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Meilleur modèle sauvegardé: {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping à l'époque {epoch+1}")
                break
        
        logger.info("Entraînement terminé")


if __name__ == "__main__":
    # Test du modèle
    logger.info("Test du modèle CV hybride")
    
    # Créer le modèle
    model = HybridCVModel(num_classes=5, gabarit_features_dim=9)
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_features = torch.randn(batch_size, 9)
    
    output = model(dummy_images, dummy_features)
    logger.info(f"Output shape: {output.shape}")  # [4, 5]
    
    # Test classificateur
    classifier = CVClassifier()
    logger.info("CVClassifier prêt à l'emploi")
