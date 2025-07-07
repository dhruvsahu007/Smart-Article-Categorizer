import numpy as np
from typing import List, Dict
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import joblib
import os

from .base_embedder import BaseEmbedder

class Word2VecEmbedder(BaseEmbedder):
    """Word2Vec embedder that averages word vectors for document representation."""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4):
        super().__init__("Word2Vec")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        
    def _preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """Preprocess texts for Word2Vec training."""
        return [simple_preprocess(text) for text in texts]
    
    def fit(self, texts: List[str]) -> None:
        """Train Word2Vec model on texts."""
        print(f"Training {self.model_name} on {len(texts)} texts...")
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=processed_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10
        )
        
        self.embedding_dim = self.vector_size
        self.is_trained = True
        print(f"Training complete. Vocabulary size: {len(self.model.wv.key_to_index)}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to averaged word embeddings."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        embeddings = []
        for text in texts:
            words = simple_preprocess(text)
            
            # Get embeddings for words in vocabulary
            word_vectors = []
            for word in words:
                if word in self.model.wv.key_to_index:
                    word_vectors.append(self.model.wv[word])
            
            # Average word vectors, or use zero vector if no words found
            if word_vectors:
                embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(self.vector_size)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_model_data(self) -> dict:
        """Get model-specific data for saving."""
        return {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'workers': self.workers,
            'model': self.model
        }
    
    def _load_model_data(self, data: dict) -> None:
        """Load model-specific data."""
        self.vector_size = data['vector_size']
        self.window = data['window']
        self.min_count = data['min_count']
        self.workers = data['workers']
        self.model = data['model']
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get vector for a specific word."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if word in self.model.wv.key_to_index:
            return self.model.wv[word]
        else:
            return np.zeros(self.vector_size)
    
    def get_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """Get similar words for a given word."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if word in self.model.wv.key_to_index:
            return self.model.wv.most_similar(word, topn=topn)
        else:
            return [] 