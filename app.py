import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Any

# Import our custom modules
from config import CATEGORIES, UI_TITLE, UI_DESCRIPTION, MODEL_CONFIGS
from data.sample_data import SampleDataGenerator
from models.classifier import ArticleClassifier
from utils.visualization import EmbeddingVisualizer
from utils.text_preprocessing import TextPreprocessor

# Page configuration
st.set_page_config(
    page_title=UI_TITLE,
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = EmbeddingVisualizer(CATEGORIES)

def load_sample_data():
    """Load or generate sample data."""
    if st.session_state.sample_data is None:
        with st.spinner("Generating sample data..."):
            generator = SampleDataGenerator()
            st.session_state.sample_data = generator.generate_dataset(samples_per_category=15)
    return st.session_state.sample_data

def train_models():
    """Train all embedding models and classifiers."""
    try:
        # Load data
        df = load_sample_data()
        
        # Initialize classifier
        st.session_state.classifier = ArticleClassifier(CATEGORIES)
        
        # Prepare data
        with st.spinner("Preparing training data..."):
            X_train, X_test, y_train, y_test = st.session_state.classifier.prepare_data(
                df['text'].tolist(), df['category'].tolist()
            )
        
        # Train embedders
        with st.spinner("Training embedding models... This may take a few minutes."):
            st.session_state.classifier.train_embedders(X_train)
        
        # Train classifiers
        with st.spinner("Training classifiers..."):
            st.session_state.classifier.train_classifiers()
        
        # Evaluate models
        with st.spinner("Evaluating models..."):
            st.session_state.classifier.evaluate_models()
        
        st.session_state.models_trained = True
        st.success("✅ All models trained successfully!")
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        st.session_state.models_trained = False

def main():
    """Main application function."""
    
    # Title and description
    st.title(UI_TITLE)
    st.markdown(UI_DESCRIPTION)
    
    # Sidebar
    st.sidebar.header("Navigation")
    
    # Check if models are trained
    if not st.session_state.models_trained:
        st.sidebar.warning("⚠️ Models not trained yet!")
        if st.sidebar.button("🚀 Train Models", type="primary"):
            train_models()
    else:
        st.sidebar.success("✅ Models trained and ready!")
    
    # Navigation options
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Model Training", "🔍 Article Classification", "📈 Performance Analysis", "🎯 Embedding Visualization"]
    )
    
    # OpenAI API Key input
    st.sidebar.header("OpenAI Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Required for OpenAI embeddings")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Model Training":
        show_training_page()
    elif page == "🔍 Article Classification":
        show_classification_page()
    elif page == "📈 Performance Analysis":
        show_performance_page()
    elif page == "🎯 Embedding Visualization":
        show_visualization_page()

def show_home_page():
    """Show the home page."""
    st.header("Welcome to Smart Article Categorizer")
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.write("- **4 Embedding Models**: Word2Vec, BERT, Sentence-BERT, OpenAI")
        st.write("- **6 Categories**: Tech, Finance, Healthcare, Sports, Politics, Entertainment")
        st.write("- **Real-time Classification**: Instant predictions with confidence scores")
        st.write("- **Performance Comparison**: Detailed metrics and visualizations")
        st.write("- **Embedding Analysis**: Cluster visualization and statistics")
    
    with col2:
        st.subheader("🚀 Getting Started")
        st.write("1. **Train Models**: Click 'Train Models' in the sidebar")
        st.write("2. **Classify Articles**: Enter text to get predictions")
        st.write("3. **Analyze Performance**: Compare model accuracy and metrics")
        st.write("4. **Visualize Embeddings**: Explore clustering patterns")
        st.write("5. **OpenAI Integration**: Add API key for OpenAI embeddings")
    
    # Sample data preview
    st.subheader("📝 Sample Data")
    df = load_sample_data()
    st.dataframe(df.head(10))
    
    # Category distribution
    st.subheader("📊 Category Distribution")
    category_counts = df['category'].value_counts()
    st.bar_chart(category_counts)

def show_training_page():
    """Show the model training page."""
    st.header("📊 Model Training")
    
    # Training controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Model selection
        st.write("**Available Models:**")
        for model_name, config in MODEL_CONFIGS.items():
            st.write(f"- **{config['name']}**: {config['description']}")
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🚀 Train All Models", type="primary"):
            train_models()
        
        if st.button("🔄 Retrain Models"):
            st.session_state.models_trained = False
            st.session_state.classifier = None
            train_models()
    
    # Show training progress/results
    if st.session_state.models_trained and st.session_state.classifier:
        st.subheader("✅ Training Complete")
        
        # Show basic results
        results = st.session_state.classifier.results
        if results:
            summary_df = st.session_state.classifier.get_performance_summary()
            st.dataframe(summary_df)
    
    # Training data info
    st.subheader("📈 Training Data Information")
    df = load_sample_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Categories", len(df['category'].unique()))
    with col3:
        st.metric("Avg Text Length", f"{df['text'].str.len().mean():.0f} chars")

def show_classification_page():
    """Show the article classification page."""
    st.header("🔍 Article Classification")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first!")
        return
    
    # Text input
    st.subheader("Enter Article Text")
    user_text = st.text_area(
        "Article content:",
        height=200,
        placeholder="Enter the article text you want to classify..."
    )
    
    # Classification button
    if st.button("🎯 Classify Article", type="primary") and user_text:
        with st.spinner("Classifying article..."):
            try:
                # Get predictions from all models
                predictions = st.session_state.classifier.predict_single(user_text)
                
                if predictions:
                    # Display results
                    st.subheader("📊 Classification Results")
                    
                    # Create results table
                    results_data = []
                    for model_name, pred_data in predictions.items():
                        results_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Prediction': pred_data['prediction'],
                            'Confidence': f"{pred_data['confidence']:.4f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Confidence scores visualization
                    st.subheader("📈 Confidence Scores")
                    fig = st.session_state.visualizer.plot_confidence_scores(
                        predictions, user_text
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed scores
                    st.subheader("🔍 Detailed Scores by Category")
                    for model_name, pred_data in predictions.items():
                        with st.expander(f"{model_name.replace('_', ' ').title()} Scores"):
                            scores_df = pd.DataFrame(
                                list(pred_data['all_scores'].items()),
                                columns=['Category', 'Score']
                            )
                            scores_df = scores_df.sort_values('Score', ascending=False)
                            st.dataframe(scores_df)
                
                else:
                    st.error("No predictions available. Please check if models are trained properly.")
                    
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")
    
    # Sample texts for testing
    st.subheader("📝 Sample Texts for Testing")
    
    sample_texts = {
        "Tech": "Apple announced a breakthrough in artificial intelligence with their new neural processing chip that delivers unprecedented performance for machine learning applications.",
        "Finance": "The Federal Reserve decided to maintain interest rates at current levels following concerns about inflation and economic growth projections for the next quarter.",
        "Healthcare": "New research shows promising results for gene therapy treatment in patients with rare genetic disorders, offering hope for previously untreatable conditions.",
        "Sports": "The championship game ended in a thrilling overtime victory with record-breaking performance from both teams in front of a sold-out stadium.",
        "Politics": "Congressional leaders are debating new legislation on healthcare reform and its potential impact on American families and healthcare accessibility.",
        "Entertainment": "The highly anticipated movie sequel broke box office records worldwide with stunning visual effects and compelling storytelling that captivated audiences."
    }
    
    selected_category = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
    
    if st.button("Use Sample Text"):
        st.text_area("Sample text:", value=sample_texts[selected_category], height=100, key="sample_display")

def show_performance_page():
    """Show the performance analysis page."""
    st.header("📈 Performance Analysis")
    
    if not st.session_state.models_trained or not st.session_state.classifier.results:
        st.warning("⚠️ Please train the models first!")
        return
    
    results = st.session_state.classifier.results
    
    # Performance summary
    st.subheader("📊 Performance Summary")
    summary_df = st.session_state.classifier.get_performance_summary()
    st.dataframe(summary_df, use_container_width=True)
    
    # Performance comparison chart
    st.subheader("📈 Performance Comparison")
    fig = st.session_state.visualizer.plot_performance_comparison(results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model rankings
    st.subheader("🏆 Model Rankings")
    fig = st.session_state.visualizer.plot_model_rankings(results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis by model
    st.subheader("🔍 Detailed Analysis by Model")
    
    selected_model = st.selectbox("Select model for detailed analysis:", list(results.keys()))
    
    if selected_model:
        # Classification report
        st.subheader(f"Classification Report - {selected_model.replace('_', ' ').title()}")
        report = st.session_state.classifier.get_classification_report(selected_model)
        st.text(report)
        
        # Confusion matrix
        st.subheader(f"Confusion Matrix - {selected_model.replace('_', ' ').title()}")
        cm = st.session_state.classifier.get_confusion_matrix(selected_model)
        fig = st.session_state.visualizer.plot_confusion_matrix(
            cm, f"Confusion Matrix - {selected_model.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_visualization_page():
    """Show the embedding visualization page."""
    st.header("🎯 Embedding Visualization")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first!")
        return
    
    # Get sample data for visualization
    df = load_sample_data()
    texts = df['text'].tolist()
    labels = df['category'].tolist()
    
    # Model selection for visualization
    available_models = []
    for name, embedder in st.session_state.classifier.embedders.items():
        if embedder.is_trained:
            available_models.append(name)
    
    if not available_models:
        st.error("No trained models available for visualization.")
        return
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox("Select embedding model:", available_models)
    
    with col2:
        reduction_method = st.selectbox("Dimensionality reduction method:", ["UMAP", "PCA", "t-SNE"])
    
    # Generate visualization
    if st.button("🎨 Generate Visualization"):
        with st.spinner(f"Generating {reduction_method} visualization for {selected_model}..."):
            try:
                # Get embeddings
                embeddings = st.session_state.classifier.get_embeddings(texts, selected_model)
                
                # Create visualization
                fig = st.session_state.visualizer.plot_embedding_clusters(
                    embeddings, labels, reduction_method.lower(),
                    f"{selected_model.replace('_', ' ').title()} Embeddings"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Embedding statistics
                st.subheader("📊 Embedding Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Dimensions", embeddings.shape[1])
                with col2:
                    st.metric("Samples", embeddings.shape[0])
                with col3:
                    st.metric("Mean", f"{np.mean(embeddings):.4f}")
                with col4:
                    st.metric("Std Dev", f"{np.std(embeddings):.4f}")
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
    
    # Comparison across models
    st.subheader("🔄 Model Comparison")
    
    if st.button("📊 Compare All Models"):
        with st.spinner("Comparing embedding statistics across models..."):
            try:
                embeddings_dict = {}
                for model_name in available_models:
                    embeddings_dict[model_name] = st.session_state.classifier.get_embeddings(texts, model_name)
                
                # Statistics comparison
                fig = st.session_state.visualizer.plot_embedding_statistics(embeddings_dict)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error comparing models: {str(e)}")

if __name__ == "__main__":
    main() 