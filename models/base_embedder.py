from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Tuple
import joblib
import os

class BaseEmbedder(ABC):
    """Base class for all embedding models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.embedding_dim = None
        
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Train the embedding model on texts."""
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to embeddings."""
        pass
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to embeddings."""
        self.fit(texts)
        return self.transform(texts)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'embedding_dim': self.embedding_dim
        }
        
        # Add model-specific data
        model_data.update(self._get_model_data())
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.embedding_dim = model_data['embedding_dim']
        
        # Load model-specific data
        self._load_model_data(model_data)
        print(f"Model loaded from {filepath}")
    
    @abstractmethod
    def _get_model_data(self) -> dict:
        """Get model-specific data for saving."""
        pass
    
    @abstractmethod
    def _load_model_data(self, data: dict) -> None:
        """Load model-specific data."""
        pass
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        if self.embedding_dim is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.embedding_dim
    
    def __str__(self) -> str:
        return f"{self.model_name} (trained: {self.is_trained}, dim: {self.embedding_dim})" 