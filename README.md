# Smart Article Categorizer

A comprehensive article classification system that automatically categorizes articles into 6 categories using 4 different embedding approaches: Word2Vec/GloVe, BERT, Sentence-BERT, and OpenAI embeddings.

## 🎯 Features

### 🤖 4 Embedding Models
- **Word2Vec**: Custom trained word vectors with document-level averaging
- **BERT**: Pre-trained BERT model using [CLS] token embeddings
- **Sentence-BERT**: All-MiniLM-L6-v2 model for direct sentence embeddings
- **OpenAI**: text-embedding-ada-002 API for state-of-the-art embeddings

### 📊 6 Categories
- **Tech**: Technology, AI, gadgets, software
- **Finance**: Banking, cryptocurrency, markets, economics
- **Healthcare**: Medical research, treatments, health policy
- **Sports**: Games, tournaments, athletes, competitions
- **Politics**: Government, elections, policy, diplomacy
- **Entertainment**: Movies, music, celebrities, events

### 🚀 Core Capabilities
- **Real-time Classification**: Instant predictions with confidence scores
- **Performance Comparison**: Detailed metrics (accuracy, precision, recall, F1-score)
- **Embedding Visualization**: 2D cluster visualization using UMAP/PCA/t-SNE
- **Model Analysis**: Confusion matrices and classification reports
- **Interactive UI**: Streamlit-based web interface

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Smart-Article-Categorizer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (Optional)**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" >> .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

### Web Interface

1. **Launch the app**: Run `streamlit run app.py` and open your browser
2. **Train models**: Click "Train Models" in the sidebar (first time only)
3. **Classify articles**: Enter text in the classification page
4. **Analyze performance**: View detailed metrics and comparisons
5. **Visualize embeddings**: Explore clustering patterns

### Command Line Usage

```python
from models.classifier import ArticleClassifier
from data.sample_data import SampleDataGenerator

# Generate sample data
generator = SampleDataGenerator()
df = generator.generate_dataset(samples_per_category=20)

# Initialize classifier
classifier = ArticleClassifier(categories=["Tech", "Finance", "Healthcare", "Sports", "Politics", "Entertainment"])

# Train models
classifier.prepare_data(df['text'].tolist(), df['category'].tolist())
classifier.train_embedders(classifier.training_data['X_train'])
classifier.train_classifiers()

# Evaluate performance
results = classifier.evaluate_models()
print(classifier.get_performance_summary())

# Classify new text
predictions = classifier.predict_single("Apple announced a new iPhone with AI capabilities")
print(predictions)
```

## 📊 Performance Metrics

The system evaluates each embedding approach using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (weighted average)
- **Recall**: Recall score (weighted average)
- **F1-Score**: F1-score (weighted average)
- **Confusion Matrix**: Detailed error analysis
- **Classification Report**: Per-category performance

## 🎨 Visualization Features

### Embedding Clusters
- **UMAP**: Uniform Manifold Approximation and Projection
- **PCA**: Principal Component Analysis
- **t-SNE**: t-distributed Stochastic Neighbor Embedding

### Performance Charts
- Model comparison bar charts
- Confidence score visualizations
- Model ranking heatmaps
- Embedding statistics comparison

## 🔧 Technical Architecture

### Project Structure
```
Smart-Article-Categorizer/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── data/
│   └── sample_data.py         # Sample data generator
├── models/
│   ├── base_embedder.py       # Base embedding class
│   ├── word2vec_embedder.py   # Word2Vec implementation
│   ├── bert_embedder.py       # BERT implementation
│   ├── sentence_bert_embedder.py  # Sentence-BERT implementation
│   ├── openai_embedder.py     # OpenAI API integration
│   └── classifier.py          # Classification pipeline
└── utils/
    ├── text_preprocessing.py   # Text cleaning utilities
    └── visualization.py        # Visualization tools
```

### Model Pipeline
1. **Data Preparation**: Text cleaning and preprocessing
2. **Embedding Generation**: Transform text to numerical vectors
3. **Classification**: Logistic regression on embeddings
4. **Evaluation**: Performance metrics and analysis
5. **Visualization**: Embedding clusters and results

## 📈 Model Comparison

| Model | Embedding Dim | Training Time | Accuracy | Best For |
|-------|---------------|---------------|----------|----------|
| Word2Vec | 100 | Fast | Good | Small datasets |
| BERT | 768 | Slow | High | Complex text |
| Sentence-BERT | 384 | Medium | High | Semantic similarity |
| OpenAI | 1536 | Fast* | Highest | Production use |

*Requires API calls

## 🔑 OpenAI Integration

### API Key Setup
1. Get your API key from [OpenAI Platform](https://platform.openai.com/)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. Or add it in the Streamlit sidebar

### Cost Considerations
- OpenAI embeddings cost $0.0004 per 1K tokens
- Estimated cost for 1000 articles: ~$0.10-$0.50
- API rate limits apply

## 🧪 Testing

### Sample Data
The system includes built-in sample data with:
- 15 articles per category (90 total)
- Realistic content for each category
- Balanced dataset for training

### Custom Data
To use your own data:
1. Create a CSV file with 'text' and 'category' columns
2. Modify the data loading in `app.py`
3. Ensure categories match the configured ones

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

**CUDA/GPU Issues**
- Models will automatically use CPU if GPU unavailable
- Install PyTorch with CUDA support for GPU acceleration

**Memory Issues**
- Reduce batch sizes in embedding models
- Use smaller datasets for testing
- Consider using lighter models first

**API Errors**
- Check OpenAI API key validity
- Verify API rate limits and quotas
- Ensure internet connection for API calls

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and libraries
- **OpenAI**: For embedding API and models
- **Streamlit**: For the web interface framework
- **Scikit-learn**: For machine learning utilities
- **Plotly**: For interactive visualizations

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include error messages and system information

---

**Built with ❤️ for article classification and NLP research** 