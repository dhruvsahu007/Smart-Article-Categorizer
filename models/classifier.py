import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import time

from .base_embedder import BaseEmbedder
from .word2vec_embedder import Word2VecEmbedder
from .bert_embedder import BertEmbedder
from .sentence_bert_embedder import SentenceBertEmbedder
from .openai_embedder import OpenAIEmbedder

class ArticleClassifier:
    """Classification pipeline for article categorization using different embeddings."""
    
    def __init__(self, categories: List[str], test_size: float = 0.2, random_state: int = 42):
        self.categories = categories
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize embedders
        self.embedders = {
            'word2vec': Word2VecEmbedder(),
            'bert': BertEmbedder(),
            'sentence_bert': SentenceBertEmbedder(),
            'openai': OpenAIEmbedder()
        }
        
        # Initialize classifiers
        self.classifiers = {}
        
        # Storage for results
        self.results = {}
        self.training_data = None
        self.embeddings_cache = {}
        
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Prepare training and test data."""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=self.test_size, random_state=self.random_state,
            stratify=labels
        )
        
        self.training_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_embedders(self, X_train: List[str]) -> None:
        """Train all embedding models."""
        print("Training embedding models...")
        
        for name, embedder in self.embedders.items():
            print(f"\n--- Training {name} ---")
            start_time = time.time()
            
            try:
                embedder.fit(X_train)
                training_time = time.time() - start_time
                print(f"Training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                if name == 'openai':
                    print("Note: OpenAI embedder requires API key. Skipping...")
                    continue
                else:
                    raise
    
    def get_embeddings(self, texts: List[str], embedder_name: str) -> np.ndarray:
        """Get embeddings for texts using specified embedder."""
        if embedder_name not in self.embedders:
            raise ValueError(f"Unknown embedder: {embedder_name}")
        
        embedder = self.embedders[embedder_name]
        
        if not embedder.is_trained:
            raise ValueError(f"Embedder {embedder_name} not trained")
        
        return embedder.transform(texts)
    
    def train_classifiers(self) -> None:
        """Train logistic regression classifiers for each embedding type."""
        if self.training_data is None:
            raise ValueError("No training data available. Call prepare_data() first.")
        
        X_train = self.training_data['X_train']
        y_train = self.training_data['y_train']
        
        print("Training classifiers...")
        
        for name, embedder in self.embedders.items():
            if not embedder.is_trained:
                print(f"Skipping {name} - embedder not trained")
                continue
                
            print(f"\n--- Training classifier for {name} ---")
            start_time = time.time()
            
            try:
                # Get embeddings for training data
                X_train_emb = embedder.transform(X_train)
                
                # Train logistic regression classifier
                classifier = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                )
                classifier.fit(X_train_emb, y_train)
                
                self.classifiers[name] = classifier
                training_time = time.time() - start_time
                print(f"Classifier training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error training classifier for {name}: {str(e)}")
                continue
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models and return performance metrics."""
        if self.training_data is None:
            raise ValueError("No training data available. Call prepare_data() first.")
        
        X_test = self.training_data['X_test']
        y_test = self.training_data['y_test']
        
        print("Evaluating models...")
        
        results = {}
        
        for name in self.classifiers.keys():
            print(f"\n--- Evaluating {name} ---")
            
            try:
                # Get embeddings for test data
                X_test_emb = self.embedders[name].transform(X_test)
                
                # Make predictions
                y_pred = self.classifiers[name].predict(X_test_emb)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred.tolist(),
                    'true_labels': y_test
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-score: {f1:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def predict_single(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Predict category for a single text using all trained models."""
        predictions = {}
        
        for name in self.classifiers.keys():
            try:
                # Get embedding
                embedding = self.embedders[name].transform([text])
                
                # Make prediction
                prediction = self.classifiers[name].predict(embedding)[0]
                probabilities = self.classifiers[name].predict_proba(embedding)[0]
                
                # Get confidence scores for all categories
                confidence_scores = {}
                for i, category in enumerate(self.categories):
                    confidence_scores[category] = probabilities[i]
                
                predictions[name] = {
                    'prediction': prediction,
                    'confidence': max(probabilities),
                    'all_scores': confidence_scores
                }
                
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
                continue
        
        return predictions
    
    def get_classification_report(self, embedder_name: str) -> str:
        """Get detailed classification report for a specific embedder."""
        if embedder_name not in self.results:
            raise ValueError(f"No results available for {embedder_name}")
        
        y_test = self.results[embedder_name]['true_labels']
        y_pred = self.results[embedder_name]['predictions']
        
        return classification_report(y_test, y_pred, target_names=self.categories)
    
    def get_confusion_matrix(self, embedder_name: str) -> np.ndarray:
        """Get confusion matrix for a specific embedder."""
        if embedder_name not in self.results:
            raise ValueError(f"No results available for {embedder_name}")
        
        y_test = self.results[embedder_name]['true_labels']
        y_pred = self.results[embedder_name]['predictions']
        
        return confusion_matrix(y_test, y_pred)
    
    def save_models(self, save_dir: str) -> None:
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embedders
        for name, embedder in self.embedders.items():
            if embedder.is_trained:
                embedder.save_model(os.path.join(save_dir, f"{name}_embedder.pkl"))
        
        # Save classifiers
        for name, classifier in self.classifiers.items():
            joblib.dump(classifier, os.path.join(save_dir, f"{name}_classifier.pkl"))
        
        # Save results
        joblib.dump(self.results, os.path.join(save_dir, "results.pkl"))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str) -> None:
        """Load trained models."""
        # Load embedders
        for name in self.embedders.keys():
            embedder_path = os.path.join(save_dir, f"{name}_embedder.pkl")
            if os.path.exists(embedder_path):
                self.embedders[name].load_model(embedder_path)
        
        # Load classifiers
        for name in self.embedders.keys():
            classifier_path = os.path.join(save_dir, f"{name}_classifier.pkl")
            if os.path.exists(classifier_path):
                self.classifiers[name] = joblib.load(classifier_path)
        
        # Load results
        results_path = os.path.join(save_dir, "results.pkl")
        if os.path.exists(results_path):
            self.results = joblib.load(results_path)
        
        print(f"Models loaded from {save_dir}")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame."""
        if not self.results:
            raise ValueError("No results available. Run evaluate_models() first.")
        
        summary_data = []
        for name, metrics in self.results.items():
            summary_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('F1-Score', ascending=False) 