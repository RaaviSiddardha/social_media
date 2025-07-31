# Social Media Analytics: Whispers of War in the Digital Sphere

## 📊 Project Overview

This project analyzes social media content related to conflicts and geopolitical events, specifically focusing on the Russia-Ukraine conflict. The analysis includes sentiment analysis, engagement prediction, entity extraction, and network analysis to understand how conflicts are discussed and spread across digital platforms.

## 🎯 Key Features

- **Sentiment Analysis**: Multi-method sentiment analysis using VADER and TextBlob
- **Engagement Prediction**: Machine learning model to predict high-engagement posts
- **Entity Extraction**: Named entity recognition to identify key players and locations
- **Network Analysis**: Graph-based analysis of entity relationships
- **Data Visualization**: Comprehensive visualizations of trends and patterns
- **Hyperparameter Tuning**: Optimized Random Forest model with GridSearchCV

## 📈 Analysis Results

- **Model Accuracy**: 95% accuracy in predicting high-engagement posts
- **Dataset Size**: 2,546 social media posts
- **Platforms Analyzed**: YouTube, TikTok, and other social media platforms
- **Time Period**: Covers recent conflict-related content

## 🛠️ Technologies Used

- **Python Libraries**:
  - `pandas` & `numpy` - Data manipulation and analysis
  - `matplotlib` & `seaborn` - Data visualization
  - `nltk` & `spacy` - Natural language processing
  - `textblob` & `vaderSentiment` - Sentiment analysis
  - `scikit-learn` - Machine learning models
  - `networkx` - Network analysis and visualization

- **Machine Learning**:
  - Random Forest Classifier
  - GridSearchCV for hyperparameter optimization
  - Feature importance analysis
  - ROC curve analysis

## 📁 Project Structure

```
social_media/
├── social_media.ipynb          # Main analysis notebook
├── conflicts_hashtag_search.csv # Dataset
├── README.md                   # This file
└── word-cloud.png             # Generated visualizations
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn nltk spacy textblob vaderSentiment scikit-learn networkx
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RaaviSiddardha/social_media.git
cd social_media
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### Running the Analysis

1. Open the Jupyter notebook:
```bash
jupyter notebook social_media.ipynb
```

2. Run all cells to perform the complete analysis

## 📊 Key Findings

### Sentiment Analysis
- Multi-method sentiment analysis reveals public opinion trends
- VADER and TextBlob provide complementary sentiment insights
- Sentiment distribution across different platforms and time periods

### Engagement Prediction
- **95% accuracy** in predicting high-engagement posts
- Key features: sentiment scores and post length
- Model can identify content likely to go viral

### Entity Network Analysis
- Identifies key entities (people, organizations, locations) in conflict discussions
- Network visualization shows relationships between mentioned entities
- Reveals central figures and organizations in the discourse

### Engagement Patterns
- Analysis of likes, comments, and views over time
- Identification of peak engagement periods
- Understanding of what drives social media engagement

## 🔍 Methodology

1. **Data Preprocessing**:
   - Text cleaning and normalization
   - Stop word removal and lemmatization
   - Handling missing values

2. **Feature Engineering**:
   - Sentiment scores (VADER and TextBlob)
   - Post length analysis
   - Entity extraction

3. **Model Development**:
   - Random Forest Classifier
   - Hyperparameter optimization with GridSearchCV
   - Cross-validation for robust evaluation

4. **Evaluation Metrics**:
   - Accuracy, precision, recall, F1-score
   - ROC curve and AUC analysis
   - Confusion matrix visualization

## 📈 Visualizations

The project generates several key visualizations:
- Sentiment distribution across posts
- Engagement metrics over time
- Entity relationship networks
- Feature importance plots
- ROC curves for model evaluation
- Confusion matrices

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Raavi Siddardha**
- GitHub: [@RaaviSiddardha](https://github.com/RaaviSiddardha)

## 🙏 Acknowledgments

- Data sources: Social media platforms (YouTube, TikTok, etc.)
- Libraries: NLTK, spaCy, scikit-learn, and the open-source community
- Research inspiration: Digital conflict analysis and social media studies

---

**Note**: This project analyzes publicly available social media data for research purposes. All analysis is conducted in accordance with ethical guidelines for social media research. 
