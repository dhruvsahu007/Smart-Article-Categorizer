import numpy as np
from typing import List, Dict
import torch
from transformers import BertTokenizer, BertModel
import joblib
import os

from .base_embedder import BaseEmbedder

class BertEmbedder(BaseEmbedder):
    """BERT embedder using [CLS] token embeddings for document representation."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        super().__init__("BERT")
        self.bert_model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, texts: List[str]) -> None:
        """Load pre-trained BERT model and tokenizer."""
        print(f"Loading {self.model_name} model ({self.bert_model_name})...")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.model = BertModel.from_pretrained(self.bert_model_name)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Set embedding dimension (BERT-base has 768 dimensions)
        self.embedding_dim = self.model.config.hidden_size
        self.is_trained = True
        
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        print(f"Using device: {self.device}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to BERT [CLS] embeddings."""
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        embeddings = []
        
        # Process texts in batches for efficiency
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts."""
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embeddings (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return cls_embeddings.tolist()
    
    def _get_model_data(self) -> dict:
        """Get model-specific data for saving."""
        return {
            'bert_model_name': self.bert_model_name,
            'max_length': self.max_length,
            'device': str(self.device)
        }
    
    def _load_model_data(self, data: dict) -> None:
        """Load model-specific data."""
        self.bert_model_name = data['bert_model_name']
        self.max_length = data['max_length']
        
        # Re-initialize model after loading
        self.fit([])  # Empty list since we're just loading the pre-trained model
    
    def get_attention_weights(self, text: str) -> np.ndarray:
        """Get attention weights for a text (for analysis)."""
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            # Return attention weights from the last layer
            attention_weights = outputs.attentions[-1].cpu().numpy()
        
        return attention_weights 