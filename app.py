"""
Professional Twitter Sentiment Analysis System - Streamlit App
============================================================

This app provides a high-accuracy sentiment analysis of Twitter profiles and hashtags
using state-of-the-art models and optimized processing techniques.
"""

# Standard library imports
import os
import re
import time
import datetime
import gc
import warnings

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import preprocessor as tweet_preprocessor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Transformer and ML imports
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed from expanded to collapsed
)

# --- Helper Functions ---

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = tweet_preprocessor.clean(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text):
    """Extract features from text"""
    if not isinstance(text, str) or text == "":
        return {
            'vader_neg': 0.0, 'vader_neu': 1.0, 'vader_pos': 0.0, 'vader_compound': 0.0,
            'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0,
            'char_count': 0, 'word_count': 0, 'avg_word_length': 0.0
        }
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text)
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity
    char_count = len(text)
    word_count = len(text.split())
    avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
    return {
        'vader_neg': vader_scores['neg'], 'vader_neu': vader_scores['neu'], 'vader_pos': vader_scores['pos'],
        'vader_compound': vader_scores['compound'], 'textblob_polarity': textblob_polarity,
        'textblob_subjectivity': textblob_subjectivity, 'char_count': char_count,
        'word_count': word_count, 'avg_word_length': avg_word_length
    }

# --- Cached Resource Loading ---
# This function is run once and cached, making the app fast on subsequent runs.

@st.cache_resource(show_spinner="Loading AI models... This may take a minute.")
def load_models():
    """Load all pre-trained models and vectorizers."""
    device = 0 if torch.cuda.is_available() else -1

    # 1. Load Classical ML Model and Vectorizer
    try:
        classical_model = joblib.load('models/best_classical_model.pkl')
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model_status = {"classical": "✅ Classical model loaded"}
    except FileNotFoundError:
        model_status = {"classical": "❌ Model files not found! Please run the training script first to generate `models/best_classical_model.pkl` and `models/tfidf_vectorizer.pkl`."}
        return None, None, None, model_status

    # 2. Load Transformer Models
    model_names = [
        'distilbert-base-uncased-finetuned-sst-2-english',
        'cardiffnlp/twitter-roberta-base-sentiment-latest',
    ]
    transformer_pipelines = {}
    for model_name in model_names:
        try:
            pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                framework="pt"
            )
            transformer_pipelines[model_name] = pipe
            model_status[model_name] = f"✅ {model_name.split('/')[-1]} loaded"
        except Exception as e:
            model_status[model_name] = f"⚠️ Could not load {model_name}: {e}"

    return classical_model, tfidf_vectorizer, transformer_pipelines, model_status

# --- Sentiment Analysis Functions ---

def predict_with_classical(text, model, vectorizer):
    """Predict sentiment using the classical model."""
    clean_text = preprocess_text(text)
    features = extract_features(clean_text)
    features_df = pd.DataFrame([features])
    text_tfidf = vectorizer.transform([clean_text])
    combined_features = np.hstack((text_tfidf.toarray(), features_df.values))
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    confidence = probabilities[prediction]
    return prediction, confidence

def predict_with_transformer(text, model_name, pipelines):
    """Predict sentiment using a transformer pipeline."""
    pipeline_obj = pipelines.get(model_name)
    if not pipeline_obj:
        return None, None # Model not loaded
    result = pipeline_obj(text)[0]
    label = result['label']
    score = result['score']
    sentiment = 0 if label.lower() in ['negative', 'neg', 'label_0'] else 1
    return sentiment, score

def analyze_sentiment_ensemble(text, classical_model, tfidf_vectorizer, transformer_pipelines):
    """Analyze sentiment using an ensemble of models."""
    # Get predictions from transformer models
    transformer_preds = [predict_with_transformer(text, name, transformer_pipelines) for name in transformer_pipelines]
    transformer_preds = [p for p in transformer_preds if p[0] is not None] # Filter out failed predictions

    if not transformer_preds:
        # Fallback to classical if transformers fail
        return predict_with_classical(text, classical_model, tfidf_vectorizer)

    # Get prediction from classical model
    classical_sentiment, classical_confidence = predict_with_classical(text, classical_model, tfidf_vectorizer)

    # Calculate ensemble prediction (weighted average)
    transformer_weight = 0.7 / len(transformer_preds)
    classical_weight = 0.3

    weighted_sentiment = sum(s * c * transformer_weight for s, c in transformer_preds) + \
                        classical_sentiment * classical_confidence * classical_weight

    ensemble_sentiment = 1 if weighted_sentiment >= 0.5 else 0
    ensemble_confidence = (sum(c for _, c in transformer_preds) * transformer_weight + classical_confidence * classical_weight)

    return ensemble_sentiment, ensemble_confidence

# --- Data Collection Function ---

def fetch_tweets(query, max_tweets, days_ago):
    """Fetch tweets with API and fallback to sample data."""
    try:
        import tweepy
        bearer_token = "AAAAAAAAAAAAAAAAAAAAAIoF3wEAAAAAkKhBzYMZmegOlBZp0f075MvBEiA%3DiRAcUZut1As19bvpKVSzDVe4NXcjlM3lyDTurx54FvXqpPQuaD"
        client = tweepy.Client(bearer_token)
        start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
        
        # Search query needs to be formatted for the API
        search_query = f"#{query.replace('#', '')} lang:en" if query.startswith('#') else f"from:{query.replace('@', '')} lang:en"

        tweets = client.search_recent_tweets(
            query=search_query,
            max_results=min(max_tweets, 100),
            tweet_fields=['created_at', 'public_metrics', 'author_id'],
            start_time=start_time
        )

        if tweets.data:
            tweets_data = []
            for tweet in tweets.data:
                tweets_data.append({
                    'id': tweet.id, 'date': tweet.created_at, 'text': tweet.text,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'replies': tweet.public_metrics.get('reply_count', 0),
                    'quotes': tweet.public_metrics.get('quote_count', 0),
                })
            return pd.DataFrame(tweets_data)
    except Exception as e:
        st.warning(f"Could not fetch tweets from API: {e}. Using sample data for demonstration.")
    
    # Fallback to creating sample data
    st.info("Creating sample tweets for demonstration purposes.")
    positive_templates = [f"I love {query}!", f"{query} is amazing!", f"Just had a great experience with {query}!"]
    negative_templates = [f"I hate {query}!", f"{query} is terrible!", f"Just had a bad experience with {query}!"]
    
    tweets_data = []
    for i in range(max_tweets):
        template = np.random.choice(positive_templates) if i % 2 == 0 else np.random.choice(negative_templates)
        date = datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, days_ago))
        tweets_data.append({
            'id': i, 'date': date, 'text': template,
            'likes': np.random.randint(0, 1000), 'retweets': np.random.randint(0, 500),
            'replies': np.random.randint(0, 100), 'quotes': np.random.randint(0, 50),
        })
    return pd.DataFrame(tweets_data)

# --- Visualization Class ---

class SentimentVisualizer:
    """Class for visualizing sentiment analysis results."""
    def __init__(self):
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})

    def plot_sentiment_distribution(self, df, title):
        """Plot sentiment distribution"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        ax.pie(
            sentiment_counts, labels=['Negative', 'Positive'], autopct='%1.1f%%',
            startangle=90, colors=['#E0245E', '#1DA1F2'], explode=(0.05, 0.05)
        )
        ax.set_title(title, fontsize=16, fontweight='bold')
        return fig

    def plot_sentiment_over_time(self, df, title):
        """Plot sentiment over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        df['date'] = pd.to_datetime(df['date']).dt.date
        sentiment_by_date = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        sentiment_by_date.plot(kind='line', ax=ax, marker='o')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tweets')
        ax.legend(['Negative', 'Positive'], title='Sentiment')
        ax.grid(True, alpha=0.3)
        return fig

    def plot_engagement_by_sentiment(self, df, title):
        """Plot engagement metrics by sentiment"""
        fig, ax = plt.subplots(figsize=(12, 8))
        engagement_by_sentiment = df.groupby('sentiment')[['likes', 'retweets', 'replies', 'quotes']].mean().reset_index()
        engagement_melted = pd.melt(
            engagement_by_sentiment, id_vars='sentiment',
            value_vars=['likes', 'retweets', 'replies', 'quotes'],
            var_name='metric', value_name='value'
        )
        sns.barplot(x='metric', y='value', hue='sentiment', data=engagement_melted, palette=['#E0245E', '#1DA1F2'], ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Engagement Metric')
        ax.set_ylabel('Average Count')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Negative', 'Positive'], title='Sentiment')
        return fig

    def create_wordcloud(self, df, sentiment, title):
        """Create word cloud for specific sentiment"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['clean_text'])
        if not sentiment_text.strip():
             ax.text(0.5, 0.5, f"No {'Positive' if sentiment==1 else 'Negative'} tweets to display.", ha='center', va='center', fontsize=16)
             ax.axis('off')
             return fig

        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            max_words=100, colormap='Blues' if sentiment == 1 else 'Reds'
        ).generate(sentiment_text)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        return fig

    def plot_confidence_distribution(self, df, title):
        """Plot confidence distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=df, x='confidence', hue='sentiment', bins=20, alpha=0.7, kde=True, palette=['#E0245E', '#1DA1F2'], ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Negative', 'Positive'], title='Sentiment')
        ax.grid(True, alpha=0.3)
        return fig

# --- Main Streamlit App ---

def main():
    st.title("🐦 Professional Twitter Sentiment Analyzer")
    st.markdown("Analyze the sentiment of any Twitter profile or hashtag using an ensemble of state-of-the-art AI models.")

    # Load models (cached)
    classical_model, tfidf_vectorizer, transformer_pipelines, model_status = load_models()
    if not all([classical_model, tfidf_vectorizer, transformer_pipelines]):
        st.error(model_status.get("classical", "Error loading models"))
        st.stop() # Stop if models failed to load

    # --- Main Page for User Input ---
    # Create a card-style container for inputs
    st.markdown("""
    <style>
    .input-card {
        background-color: white;
        padding: 2px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .input-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #1DA1F2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-header">Analysis Configuration</div>', unsafe_allow_html=True)
        
        # Use columns for wider screens, vertical for mobile
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analysis_type = st.radio("Analyze a:", ('Profile', 'Hashtag'))
            query_input = st.text_input(
                "Enter Twitter username or hashtag",
                value="elonmusk",
                help="For profiles, enter the username (e.g., elonmusk). For hashtags, include the # (e.g., #AI)."
            )
        
        with col2:
            max_tweets = st.slider("Maximum number of tweets to fetch", 10, 200, 50)
            days_ago = st.slider("Tweets from how many days ago?", 1, 30, 7)
        
        # Center the analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display model loading status
    if model_status:
        st.markdown("### Model Status")
        for model, status in model_status.items():
            if "✅" in status:
                st.success(status, icon="🤖")
            elif "⚠️" in status:
                st.warning(status)
            else:
                st.error(status)

    if analyze_button:
        if not query_input:
            st.error("Please enter a username or hashtag.")
            return

        # Format query
        if analysis_type == 'Profile':
            query = query_input.lstrip('@')
        else: # Hashtag
            query = query_input if query_input.startswith('#') else f"#{query_input}"
        
        # --- Main Analysis Logic ---
        with st.spinner(f"Fetching and analyzing tweets for '{query}'..."):
            tweets_df = fetch_tweets(query, max_tweets, days_ago)

            if tweets_df.empty:
                st.error("No tweets found. Please check the input and try again.")
                return

            # Preprocess and analyze
            tweets_df['clean_text'] = tweets_df['text'].apply(preprocess_text)
            sentiments = []
            confidences = []
            for text in tweets_df['text']:
                sentiment, confidence = analyze_sentiment_ensemble(text, classical_model, tfidf_vectorizer, transformer_pipelines)
                sentiments.append(sentiment)
                confidences.append(confidence)
            
            tweets_df['sentiment'] = sentiments
            tweets_df['confidence'] = confidences

        # --- Display Results ---
        st.success(f"Analysis complete for {len(tweets_df)} tweets!")
        
        # Summary Statistics
        st.header("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", len(tweets_df))
        col2.metric("Positive Tweets", len(tweets_df[tweets_df['sentiment'] == 1]))
        col3.metric("Negative Tweets", len(tweets_df[tweets_df['sentiment'] == 0]))
        col4.metric("Avg. Confidence", f"{tweets_df['confidence'].mean():.2%}")

        # Sample Tweets
        st.header("📝 Sample Tweets")
        positive_tweets = tweets_df[tweets_df['sentiment'] == 1].sort_values('confidence', ascending=False).head(3)
        negative_tweets = tweets_df[tweets_df['sentiment'] == 0].sort_values('confidence', ascending=False).head(3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Positive Tweets")
            for _, row in positive_tweets.iterrows():
                st.info(f"**Confidence:** {row['confidence']:.2%}\n\n{row['text']}")
        with col2:
            st.subheader("Most Negative Tweets")
            for _, row in negative_tweets.iterrows():
                st.error(f"**Confidence:** {row['confidence']:.2%}\n\n{row['text']}")
        
        # Visualizations
        st.header("📈 Visualizations")
        visualizer = SentimentVisualizer()
        
        # Use columns for layout
        col1, col2 = st.columns(2)
        with col1:
            fig1 = visualizer.plot_sentiment_distribution(tweets_df, f"Sentiment Distribution for {query}")
            st.pyplot(fig1)
        with col2:
            fig2 = visualizer.plot_confidence_distribution(tweets_df, f"Confidence Distribution for {query}")
            st.pyplot(fig2)
        
        fig3 = visualizer.plot_sentiment_over_time(tweets_df, f"Sentiment Over Time for {query}")
        st.pyplot(fig3)
        
        fig4 = visualizer.plot_engagement_by_sentiment(tweets_df, f"Engagement by Sentiment for {query}")
        st.pyplot(fig4)

        col1, col2 = st.columns(2)
        with col1:
            fig5 = visualizer.create_wordcloud(tweets_df, sentiment=1, title=f"Positive Words for {query}")
            st.pyplot(fig5)
        with col2:
            fig6 = visualizer.create_wordcloud(tweets_df, sentiment=0, title=f"Negative Words for {query}")
            st.pyplot(fig6)
            
        # Add a new plot for confidence distribution
        def plot_confidence_distribution(self, df, title):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(data=df, x='confidence', hue='sentiment', bins=20, alpha=0.7, kde=True, palette=['#E0245E', '#1DA1F2'], ax=ax)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['Negative', 'Positive'], title='Sentiment')
            ax.grid(True, alpha=0.3)
            return fig
        
        # Monkey patch the new method into the class
        SentimentVisualizer.plot_confidence_distribution = plot_confidence_distribution


if __name__ == "__main__":
    main()
