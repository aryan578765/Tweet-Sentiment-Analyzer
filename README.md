---

# 🐦 Professional Twitter Sentiment Analyzer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌐 Live Demo

Experience the live application here: **[https://tweets-sentiment-intelligence.streamlit.app/](https://tweets-sentiment-intelligence.streamlit.app/)**

A powerful and interactive web application built with Streamlit to perform real-time sentiment analysis on Twitter profiles and hashtags. This tool leverages an ensemble of state-of-the-art transformer models and classical machine learning algorithms to provide highly accurate sentiment insights, complete with comprehensive visualizations.

<!-- 
**PRO-TIP:** Add a screenshot or GIF of your app here! It makes the README much more engaging.
Example:
![App Screenshot](images/screenshot.png)
-->

## ✨ Key Features

-   **🤖 Ensemble Model Accuracy**: Combines predictions from multiple transformer models (DistilBERT, RoBERTa) and a classical ML model (Logistic Regression) for robust sentiment analysis.
-   **🎯 Interactive UI**: A clean and intuitive Streamlit interface for easy configuration and analysis.
-   **📊 Comprehensive Visualizations**: Generates a suite of plots including sentiment distribution, sentiment over time, engagement metrics, and word clouds.
-   **🔄 Real-time Data Fetching**: Fetches live tweets using the Twitter API v2, with a fallback to sample data for demonstration.
-   **⚡ Performance Optimized**: Uses Streamlit's caching to load models only once, ensuring a fast and responsive user experience.
-   **🔒 Secure Credential Management**: Securely handles API keys using Streamlit's secrets management.

## 🛠️ Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Backend**: Python
-   **Machine Learning**:
    -   [Transformers (Hugging Face)](https://huggingface.co/): `distilbert-base-uncased-finetuned-sst-2-english`, `cardiffnlp/twitter-roberta-base-sentiment-latest`
    -   [Scikit-learn](https://scikit-learn.org/): `LogisticRegression`, `TfidfVectorizer`
-   **Data Fetching**: [Tweepy](https://www.tweepy.org/) (Twitter API v2)
-   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [WordCloud](https://amueller.github.io/word_cloud/)

## 🚀 Installation & Setup

Follow these steps to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   **Python 3.9+**: Make sure you have Python installed. You can check with `python --version`.
-   **Pip**: Usually comes with Python.
-   **Twitter Bearer Token**: You need a Twitter Developer account to get a Bearer Token for the API.
    1.  Apply for a Twitter Developer account at [https://developer.twitter.com/](https://developer.twitter.com/).
    2.  Create a new Project and a new App within that project.
    3.  Generate your "Bearer Token" from the App's "Keys and tokens" section.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/twitter-sentiment-analyzer.git
cd twitter-sentiment-analyzer
```

### 2. Create a Virtual Environment (Recommended)

It's best practice to create a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Generate Model Files

This application requires pre-trained classical ML models to function. You need to generate them by running the training part of the original notebook.

1.  Open the `Twitter_Sentiment_Analysis.ipynb` notebook in Google Colab or a local Jupyter environment.
2.  Execute cells **1 through 8**. This will train the models and save them as `best_classical_model.pkl` and `tfidf_vectorizer.pkl`.
3.  Download these two files from the Colab file browser (or find them in your local `models/` directory).
4.  Place them inside the `models/` folder of this project.

Your project structure should now look like this:

```
twitter-sentiment-analyzer/
├── models/
│   ├── best_classical_model.pkl
│   └── tfidf_vectorizer.pkl
├── .streamlit/
│   └── secrets.toml
├── app.py
├── requirements.txt
└── README.md
```

### 5. Configure API Keys

For security, API keys are managed using Streamlit's secrets.

-   **For Local Development**:
    1.  Create a folder named `.streamlit` in your project root.
    2.  Inside `.streamlit`, create a file named `secrets.toml`.
    3.  Add your credentials to `secrets.toml` like this:

    ```toml
    # .streamlit/secrets.toml

    [twitter]
    bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
    ```

-   **For Deployment on Streamlit Community Cloud**:
    -   In your app's dashboard on Streamlit Cloud, go to **Settings > Secrets**.
    -   Add the same content as above in the secrets editor.

## 🏃 Running the Application

Once the setup is complete, you can run the app from your terminal:

```bash
streamlit run app.py
```

The application will open automatically in your web browser, usually at `http://localhost:8501`.

## 📖 Usage Guide

1.  **Configure Analysis**: Use the sidebar on the left to configure your analysis.
    -   **Analyze a**: Choose between `Profile` (for a Twitter user) or `Hashtag`.
    -   **Enter Input**: Type the Twitter username (e.g., `elonmusk`) or hashtag (e.g., `#AI`).
    -   **Adjust Sliders**: Set the maximum number of tweets to fetch and the time frame.
2.  **Start Analysis**: Click the **"🔍 Analyze Sentiment"** button.
3.  **View Results**: The app will fetch, process, and analyze the tweets, then display:
    -   **Summary Statistics**: Key metrics like total tweets and average confidence.
    -   **Sample Tweets**: The most positive and negative tweets identified by the model.
    -   **Visualizations**: A series of interactive charts and graphs to explore the sentiment data.

## 📁 Project Structure

```
twitter-sentiment-analyzer/
├── .streamlit/              # Configuration folder for secrets
│   └── secrets.toml         # Stores API keys (DO NOT commit to Git)
├── models/                  # Folder for pre-trained ML models
│   ├── best_classical_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py                   # Main Streamlit application script
├── requirements.txt         # List of Python dependencies
├── README.md               # This file
└── Twitter_Sentiment_Analysis.ipynb # Original Jupyter Notebook for training
```

## 🙏 Acknowledgments

-   **[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)**: Used for training the classical machine learning model.
-   **[Hugging Face Transformers](https://huggingface.co/models)**: For providing the pre-trained transformer models.
-   **[Streamlit](https://streamlit.io/)**: For the amazing framework that makes building data apps easy and fun.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.