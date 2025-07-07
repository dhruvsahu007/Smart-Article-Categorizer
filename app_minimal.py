import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Any

# Import our custom modules with minimal dependencies
from config import CATEGORIES, UI_TITLE, UI_DESCRIPTION, MODEL_CONFIGS
from data.sample_data import SampleDataGenerator

# Page configuration
st.set_page_config(
    page_title=UI_TITLE,
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

def load_sample_data():
    """Load or generate sample data."""
    if st.session_state.sample_data is None:
        with st.spinner("Generating sample data..."):
            generator = SampleDataGenerator()
            st.session_state.sample_data = generator.generate_dataset(samples_per_category=15)
    return st.session_state.sample_data

def main():
    """Main application function."""
    
    # Title and description
    st.title(UI_TITLE)
    st.markdown(UI_DESCRIPTION)
    
    # Sidebar
    st.sidebar.header("Navigation")
    
    # Navigation options
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📝 Sample Data", "⚙️ Configuration"]
    )
    
    # OpenAI API Key input
    st.sidebar.header("OpenAI Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Required for OpenAI embeddings")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📝 Sample Data":
        show_data_page()
    elif page == "⚙️ Configuration":
        show_config_page()

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
        st.write("1. **Install Dependencies**: Make sure all packages are installed")
        st.write("2. **Check Sample Data**: View the built-in training data")
        st.write("3. **Configure Models**: Set up embedding models")
        st.write("4. **Train System**: Run training pipeline")
        st.write("5. **OpenAI Integration**: Add API key for OpenAI embeddings")
    
    # System status
    st.subheader("📊 System Status")
    
    # Check installed packages
    try:
        import streamlit
        st.success("✅ Streamlit: Ready")
    except ImportError:
        st.error("❌ Streamlit: Not available")
    
    try:
        import pandas
        st.success("✅ Pandas: Ready")
    except ImportError:
        st.error("❌ Pandas: Not available")
    
    try:
        import sklearn
        st.success("✅ Scikit-learn: Ready")
    except ImportError:
        st.error("❌ Scikit-learn: Not available")
    
    try:
        import transformers
        st.success("✅ Transformers: Ready")
    except ImportError:
        st.error("❌ Transformers: Not available")
    
    try:
        import sentence_transformers
        st.success("✅ Sentence-Transformers: Ready")
    except ImportError:
        st.error("❌ Sentence-Transformers: Not available")
    
    try:
        import openai
        st.success("✅ OpenAI: Ready")
    except ImportError:
        st.error("❌ OpenAI: Not available")

def show_data_page():
    """Show the sample data page."""
    st.header("📝 Sample Data")
    
    # Load and display sample data
    df = load_sample_data()
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Categories", len(df['category'].unique()))
    with col3:
        st.metric("Avg Text Length", f"{df['text'].str.len().mean():.0f} chars")
    
    # Category distribution
    st.subheader("📊 Category Distribution")
    category_counts = df['category'].value_counts()
    st.bar_chart(category_counts)
    
    # Sample data table
    st.subheader("📋 Sample Articles")
    
    # Category filter
    selected_category = st.selectbox("Filter by category:", ["All"] + list(df['category'].unique()))
    
    if selected_category != "All":
        filtered_df = df[df['category'] == selected_category]
    else:
        filtered_df = df
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Text length analysis
    st.subheader("📏 Text Length Analysis")
    text_lengths = df['text'].str.len()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Min Length", f"{text_lengths.min()} chars")
        st.metric("Max Length", f"{text_lengths.max()} chars")
    with col2:
        st.metric("Mean Length", f"{text_lengths.mean():.0f} chars")
        st.metric("Median Length", f"{text_lengths.median():.0f} chars")

def show_config_page():
    """Show the configuration page."""
    st.header("⚙️ Configuration")
    
    # Model configurations
    st.subheader("🤖 Model Configurations")
    
    for model_name, config in MODEL_CONFIGS.items():
        with st.expander(f"{config['name']} Configuration"):
            st.write(f"**Description**: {config['description']}")
            
            if model_name == "word2vec":
                st.write(f"**Vector Size**: {config['vector_size']}")
                st.write(f"**Window**: {config['window']}")
                st.write(f"**Min Count**: {config['min_count']}")
                st.write(f"**Workers**: {config['workers']}")
            
            elif model_name == "bert":
                st.write(f"**Model Name**: {config['model_name']}")
                st.write(f"**Max Length**: {config['max_length']}")
            
            elif model_name == "sentence_bert":
                st.write(f"**Model Name**: {config['model_name']}")
            
            elif model_name == "openai":
                st.write(f"**Model Name**: {config['model_name']}")
                st.write(f"**Max Tokens**: {config['max_tokens']}")
    
    # Categories
    st.subheader("📂 Categories")
    for i, category in enumerate(CATEGORIES, 1):
        st.write(f"{i}. **{category}**")
    
    # Environment variables
    st.subheader("🔧 Environment Variables")
    
    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ OpenAI API Key is set")
    else:
        st.warning("⚠️ OpenAI API Key is not set")
        st.info("Add your OpenAI API key in the sidebar to enable OpenAI embeddings")
    
    # Next steps
    st.subheader("🚀 Next Steps")
    st.info("""
    **To complete the setup:**
    
    1. **Install Missing Dependencies**: Some packages failed to install due to compilation issues
       - Try: `pip install --only-binary=all numpy gensim nltk wordcloud umap-learn`
       - Or install Microsoft Visual C++ Build Tools
    
    2. **Test Individual Components**: 
       - Check if embedding models can be loaded
       - Verify classification pipeline works
    
    3. **Run Full Application**: Once all dependencies are installed, use the full app.py
    """)

# Simple text classification demo using only installed packages
def show_simple_classification():
    """Simple classification demo using basic methods."""
    st.header("🔍 Simple Text Classification Demo")
    
    # Load sample data
    df = load_sample_data()
    
    # Simple keyword-based classification
    st.subheader("Keyword-Based Classification")
    
    user_text = st.text_area(
        "Enter text to classify:",
        height=150,
        placeholder="Enter article text..."
    )
    
    if st.button("🎯 Classify") and user_text:
        # Simple keyword-based classification
        keywords = {
            "Tech": ["technology", "ai", "artificial intelligence", "software", "computer", "digital", "app", "iphone", "google", "microsoft", "apple"],
            "Finance": ["money", "bank", "investment", "stock", "market", "bitcoin", "financial", "economy", "dollar", "federal reserve"],
            "Healthcare": ["health", "medical", "doctor", "hospital", "treatment", "patient", "medicine", "therapy", "clinical", "disease"],
            "Sports": ["game", "team", "player", "championship", "olympic", "football", "basketball", "soccer", "tennis", "tournament"],
            "Politics": ["government", "election", "president", "congress", "policy", "vote", "campaign", "political", "senator", "law"],
            "Entertainment": ["movie", "music", "celebrity", "film", "actor", "concert", "entertainment", "show", "television", "award"]
        }
        
        text_lower = user_text.lower()
        scores = {}
        
        for category, category_keywords in keywords.items():
            score = sum(1 for keyword in category_keywords if keyword in text_lower)
            scores[category] = score
        
        # Find best match
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]
        
        if max_score > 0:
            st.success(f"**Predicted Category**: {best_category}")
            st.write("**Keyword Scores:**")
            for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {category}: {score} keywords found")
        else:
            st.warning("No clear category detected. Try adding more specific keywords.")

if __name__ == "__main__":
    main() 