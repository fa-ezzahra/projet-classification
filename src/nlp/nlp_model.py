"""
Module NLP pour classification de texte
Utilise CamemBERT fine-tuné sur textes administratifs
"""
import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertTokenizer
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CamembertClassifier(nn.Module):
    """
    Classificateur basé sur CamemBERT
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        """
        Initialise le classificateur
        
        Args:
            num_classes: Nombre de classes
            dropout: Taux de dropout
        """
        super(CamembertClassifier, self).__init__()
        
        # Charger CamemBERT
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        
        # Freeze les premières couches (optionnel)
        # for param in self.camembert.embeddings.parameters():
        #     param.requires_grad = False
        
        # Classification head
        hidden_size = self.camembert.config.hidden_size  # 768 pour camembert-base
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"CamembertClassifier initialisé: {num_classes} classes")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Passer par CamemBERT
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Prendre la représentation [CLS]
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        
        return logits


class NLPClassifier:
    """
    Classificateur NLP avec preprocessing
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialise le classificateur NLP
        
        Args:
            model_path: Chemin vers le modèle fine-tuné (optionnel)
            device: Device PyTorch
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        
        # Initialiser le modèle
        self.model = CamembertClassifier(num_classes=3)
        
        # Charger les poids fine-tunés si disponibles
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Modèle NLP chargé depuis {model_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle NLP: {e}")
                logger.warning("Utilisation de CamemBERT sans fine-tuning")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Classes
        self.classes = [
            "piece_identite",
            "releve_notes",
            "facture"
        ]
        
        # Paramètres
        self.max_length = 512
        
        logger.info(f"NLPClassifier initialisé sur {self.device}")
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Prétraite le texte pour le modèle
        
        Args:
            text: Texte à classifier
            
        Returns:
            Dictionnaire avec input_ids et attention_mask
        """
        # Nettoyer le texte
        text = text.strip()
        
        # Tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Prédit la classe d'un texte
        
        Args:
            text: Texte à classifier
            
        Returns:
            Dictionnaire de scores par classe
        """
        if not text or len(text.strip()) < 10:
            # Texte trop court, retourner scores uniformes
            return {cls: 0.2 for cls in self.classes}
        
        with torch.no_grad():
            # Prétraiter
            inputs = self.preprocess_text(text)
            
            # Forward pass
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            
            # Softmax
            probs = torch.softmax(logits, dim=1)
            
            # Convertir en dictionnaire
            scores = {}
            for i, class_name in enumerate(self.classes):
                scores[class_name] = probs[0, i].item()
        
        return scores
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Prédit un batch de textes
        
        Args:
            texts: Liste de textes
            
        Returns:
            Liste de dictionnaires de scores
        """
        results = []
        
        # Filtrer les textes vides
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) >= 10:
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            # Tous les textes sont trop courts
            return [{cls: 0.2 for cls in self.classes} for _ in texts]
        
        # Traiter par batch de 8
        batch_size = 8
        all_scores = [None] * len(texts)
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]
            
            # Tokenizer le batch
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                # Convertir en liste de dictionnaires
                for j in range(probs.shape[0]):
                    scores = {}
                    for k, class_name in enumerate(self.classes):
                        scores[class_name] = probs[j, k].item()
                    
                    original_idx = batch_indices[j]
                    all_scores[original_idx] = scores
        
        # Remplir les textes vides
        for i in range(len(texts)):
            if all_scores[i] is None:
                all_scores[i] = {cls: 0.2 for cls in self.classes}
        
        return all_scores


class NLPModelTrainer:
    """
    Entraîneur pour le modèle NLP
    """
    
    def __init__(self, model: CamembertClassifier, device: str = None):
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
        
        # Optimizer avec learning rates différents
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.camembert.parameters(), 'lr': 2e-5},  # Petit LR pour CamemBERT
            {'params': self.model.classifier.parameters(), 'lr': 1e-4}  # Plus grand LR pour classifier
        ])
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        logger.info("NLPModelTrainer initialisé")
    
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
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            # Déplacer sur device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Loss
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update
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
            for input_ids, attention_mask, labels in val_loader:
                # Déplacer sur device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                logits = self.model(input_ids, attention_mask)
                
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
    
    def train(self, train_loader, val_loader, epochs: int = 30,
              save_path: str = "models/nlp/camembert_classifier.pth"):
        """
        Boucle d'entraînement complète
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            epochs: Nombre d'époques
            save_path: Chemin pour sauvegarder le modèle
        """
        best_val_loss = float('inf')
        patience = 5
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
            
            # Sauvegarder le meilleur
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


# Fonction utilitaire pour créer un dataset
def create_text_dataset(texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
    """
    Crée un dataset PyTorch pour l'entraînement
    
    Args:
        texts: Liste de textes
        labels: Liste de labels (indices de classes)
        tokenizer: Tokenizer CamemBERT
        max_length: Longueur maximale
        
    Returns:
        TensorDataset
    """
    from torch.utils.data import TensorDataset
    
    # Tokenizer tous les textes
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Créer dataset
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    
    return dataset


if __name__ == "__main__":
    # Test du modèle
    logger.info("Test du modèle NLP")
    
    # Créer le modèle
    model = CamembertClassifier(num_classes=3)
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    dummy_input_ids = torch.randint(0, 32005, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)
    
    output = model(dummy_input_ids, dummy_attention_mask)
    logger.info(f"Output shape: {output.shape}")  # [4, 5]
    
    # Test classificateur
    classifier = NLPClassifier()
    
    # Test avec texte
    test_text = "Facture d'électricité ONE consommation 150 kWh montant 250 DH"
    scores = classifier.predict(test_text)
    logger.info(f"Scores: {scores}")
