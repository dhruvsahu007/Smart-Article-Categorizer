import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import joblib
import os

from .base_embedder import BaseEmbedder

class SentenceBertEmbedder(BaseEmbedder):
    """Sentence-BERT embedder using all-MiniLM-L6-v2 for direct sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__("Sentence-BERT")
        self.sbert_model_name = model_name
        self.model = None
        
    def fit(self, texts: List[str]) -> None:
        """Load pre-trained Sentence-BERT model."""
        print(f"Loading {self.model_name} model ({self.sbert_model_name})...")
        
        # Load pre-trained sentence transformer model
        self.model = SentenceTransformer(self.sbert_model_name)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.is_trained = True
        
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        print(f"Model device: {self.model.device}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to Sentence-BERT embeddings."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Convert texts to embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        return embeddings
    
    def _get_model_data(self) -> dict:
        """Get model-specific data for saving."""
        return {
            'sbert_model_name': self.sbert_model_name,
            'embedding_dim': self.embedding_dim
        }
    
    def _load_model_data(self, data: dict) -> None:
        """Load model-specific data."""
        self.sbert_model_name = data['sbert_model_name']
        
        # Re-initialize model after loading
        self.fit([])  # Empty list since we're just loading the pre-trained model
    
    def get_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Get similarity matrix for a list of texts."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get embeddings
        embeddings = self.transform(texts)
        
        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def find_most_similar(self, query: str, texts: List[str], top_k: int = 5) -> List[tuple]:
        """Find most similar texts to a query."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get query embedding
        query_embedding = self.model.encode([query])
        
        # Get text embeddings
        text_embeddings = self.model.encode(texts)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((texts[idx], similarities[idx]))
        
        return results 