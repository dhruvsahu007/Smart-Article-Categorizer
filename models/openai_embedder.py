import numpy as np
from typing import List, Dict
import openai
import time
import os
from dotenv import load_dotenv
import joblib

from .base_embedder import BaseEmbedder

# Load environment variables
load_dotenv()

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedder using text-embedding-ada-002 API."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", max_tokens: int = 8191):
        super().__init__("OpenAI")
        self.openai_model_name = model_name
        self.max_tokens = max_tokens
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        # Ada-002 embedding dimension is 1536
        self.embedding_dim = 1536
        
    def fit(self, texts: List[str]) -> None:
        """Initialize OpenAI client."""
        print(f"Initializing {self.model_name} model ({self.openai_model_name})...")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Test API connection with a simple request
        try:
            test_response = self.client.embeddings.create(
                model=self.openai_model_name,
                input=["test"]
            )
            print(f"API connection successful. Embedding dimension: {len(test_response.data[0].embedding)}")
            self.embedding_dim = len(test_response.data[0].embedding)
            self.is_trained = True
            
        except Exception as e:
            raise ValueError(f"Failed to connect to OpenAI API: {str(e)}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to OpenAI embeddings."""
        if not self.is_trained or self.client is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        embeddings = []
        
        # Process texts in batches to respect rate limits
        batch_size = 100  # OpenAI allows up to 2048 inputs per request
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
            
            # Add small delay to respect rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return np.array(embeddings)
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            # Truncate texts if they exceed max tokens
            processed_texts = []
            for text in texts:
                if len(text.split()) > self.max_tokens:
                    # Simple word-based truncation
                    words = text.split()
                    processed_text = ' '.join(words[:self.max_tokens])
                    processed_texts.append(processed_text)
                else:
                    processed_texts.append(text)
            
            # Get embeddings from OpenAI
            response = self.client.embeddings.create(
                model=self.openai_model_name,
                input=processed_texts
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dim for _ in texts]
    
    def _get_model_data(self) -> dict:
        """Get model-specific data for saving."""
        return {
            'openai_model_name': self.openai_model_name,
            'max_tokens': self.max_tokens,
            'embedding_dim': self.embedding_dim,
            'api_key_set': bool(self.api_key)
        }
    
    def _load_model_data(self, data: dict) -> None:
        """Load model-specific data."""
        self.openai_model_name = data['openai_model_name']
        self.max_tokens = data['max_tokens']
        self.embedding_dim = data['embedding_dim']
        
        # Re-initialize model after loading
        self.fit([])  # Empty list since we're just initializing the API client
    
    def get_embedding_cost(self, num_tokens: int) -> float:
        """Estimate cost for embedding generation."""
        # OpenAI pricing for text-embedding-ada-002: $0.0004 per 1K tokens
        cost_per_1k_tokens = 0.0004
        return (num_tokens / 1000) * cost_per_1k_tokens
    
    def get_text_similarity(self, text1: str, text2: str) -> float:
        """Get cosine similarity between two texts."""
        if not self.is_trained or self.client is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get embeddings for both texts
        embeddings = self.transform([text1, text2])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return similarity 