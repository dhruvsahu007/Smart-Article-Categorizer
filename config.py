import os
from typing import Dict, List

# Categories for classification
CATEGORIES = [
    "Tech",
    "Finance", 
    "Healthcare",
    "Sports",
    "Politics",
    "Entertainment"
]

# Model configurations
MODEL_CONFIGS = {
    "word2vec": {
        "name": "Word2Vec/GloVe",
        "description": "Average word vectors for document representation",
        "vector_size": 100,
        "window": 5,
        "min_count": 1,
        "workers": 4
    },
    "bert": {
        "name": "BERT",
        "description": "Using [CLS] token embeddings",
        "model_name": "bert-base-uncased",
        "max_length": 512
    },
    "sentence_bert": {
        "name": "Sentence-BERT",
        "description": "Direct sentence embeddings",
        "model_name": "all-MiniLM-L6-v2"
    },
    "openai": {
        "name": "OpenAI",
        "description": "text-embedding-ada-002 API",
        "model_name": "text-embedding-ada-002",
        "max_tokens": 8191
    }
}

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# File paths
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Training parameters
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# UI configuration
UI_TITLE = "Smart Article Categorizer"
UI_DESCRIPTION = """
This application classifies articles into 6 categories using different embedding approaches:
- **Word2Vec/GloVe**: Average word vectors
- **BERT**: [CLS] token embeddings  
- **Sentence-BERT**: Direct sentence embeddings
- **OpenAI**: text-embedding-ada-002 API
"""

# Visualization settings
PLOT_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine"
} 