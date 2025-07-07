import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import List, Dict, Tuple, Any
import streamlit as st

class EmbeddingVisualizer:
    """Visualization utilities for embedding analysis and model performance."""
    
    def __init__(self, categories: List[str], colors: List[str] = None):
        self.categories = categories
        self.colors = colors or px.colors.qualitative.Set3[:len(categories)]
        
    def plot_embedding_clusters(self, embeddings: np.ndarray, labels: List[str], 
                               method: str = 'umap', title: str = "Embedding Clusters") -> go.Figure:
        """Plot embedding clusters using dimensionality reduction."""
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            subtitle = f"PCA Projection (Explained Variance: {reducer.explained_variance_ratio_.sum():.2%})"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            reduced_embeddings = reducer.fit_transform(embeddings)
            subtitle = "t-SNE Projection"
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            subtitle = "UMAP Projection"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'category': labels
        })
        
        # Create scatter plot
        fig = px.scatter(
            df, x='x', y='y', color='category',
            title=f"{title}<br><sub>{subtitle}</sub>",
            color_discrete_sequence=self.colors,
            hover_data=['category']
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_performance_comparison(self, results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Plot performance comparison across different embedding methods."""
        
        # Prepare data
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Color scheme
        colors = px.colors.qualitative.Set2[:len(models)]
        
        # Add bar charts for each metric
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models, 
                    y=values,
                    name=metric.capitalize(),
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=600,
            showlegend=False
        )
        
        # Update y-axis range
        fig.update_yaxes(range=[0, 1])
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Confusion Matrix") -> go.Figure:
        """Plot confusion matrix as heatmap."""
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=self.categories,
            y=self.categories,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=600
        )
        
        return fig
    
    def plot_confidence_scores(self, predictions: Dict[str, Dict[str, Any]], 
                              text_preview: str = None) -> go.Figure:
        """Plot confidence scores for different models."""
        
        # Prepare data
        models = list(predictions.keys())
        data = []
        
        for model in models:
            if 'all_scores' in predictions[model]:
                scores = predictions[model]['all_scores']
                for category, score in scores.items():
                    data.append({
                        'Model': model,
                        'Category': category,
                        'Confidence': score,
                        'Predicted': predictions[model]['prediction']
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig = px.bar(
            df, x='Category', y='Confidence', color='Model',
            title="Model Confidence Scores by Category",
            barmode='group',
            height=500
        )
        
        # Add subtitle with text preview if provided
        if text_preview:
            preview = text_preview[:100] + "..." if len(text_preview) > 100 else text_preview
            fig.update_layout(
                title=f"Model Confidence Scores by Category<br><sub>Text: {preview}</sub>"
            )
        
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Confidence Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_model_rankings(self, results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Plot model rankings across different metrics."""
        
        # Prepare data
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Calculate rankings (1 = best, n = worst)
        rankings = {}
        for metric in metrics:
            values = [(model, results[model][metric]) for model in models]
            values.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = {model: rank + 1 for rank, (model, _) in enumerate(values)}
        
        # Create ranking DataFrame
        ranking_data = []
        for model in models:
            for metric in metrics:
                ranking_data.append({
                    'Model': model,
                    'Metric': metric.capitalize(),
                    'Rank': rankings[metric][model],
                    'Score': results[model][metric]
                })
        
        df = pd.DataFrame(ranking_data)
        
        # Create heatmap
        pivot_df = df.pivot(index='Model', columns='Metric', values='Rank')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlBu_r',
            text=pivot_df.values,
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Model Rankings by Metric (1 = Best)",
            xaxis_title="Metric",
            yaxis_title="Model",
            width=600,
            height=400
        )
        
        return fig
    
    def plot_embedding_statistics(self, embeddings_dict: Dict[str, np.ndarray]) -> go.Figure:
        """Plot embedding statistics comparison."""
        
        # Calculate statistics
        stats = {}
        for name, embeddings in embeddings_dict.items():
            stats[name] = {
                'mean': np.mean(embeddings),
                'std': np.std(embeddings),
                'min': np.min(embeddings),
                'max': np.max(embeddings),
                'dimension': embeddings.shape[1]
            }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean', 'Standard Deviation', 'Min Value', 'Max Value']
        )
        
        models = list(stats.keys())
        metrics = ['mean', 'std', 'min', 'max']
        colors = px.colors.qualitative.Set1[:len(models)]
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [stats[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.capitalize(),
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Embedding Statistics Comparison",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_performance_table(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a formatted performance table."""
        
        data = []
        for model, metrics in results.items():
            data.append({
                'Model': model.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df = pd.DataFrame(data)
        
        # Sort by F1-Score
        df['F1_numeric'] = [results[model]['f1_score'] for model in results.keys()]
        df = df.sort_values('F1_numeric', ascending=False)
        df = df.drop('F1_numeric', axis=1)
        
        return df 