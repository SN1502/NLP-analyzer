import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import streamlit as st

# Downloading necessary NLTK data
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.clf = None

    def preprocess_text(self, text):
        if isinstance(text, str):
            return text.lower()
        else:
            return str(text).lower()

    def train_classifier(self, reviews, labels):
        vectorizer = CountVectorizer(preprocessor=self.preprocess_text)
        X = vectorizer.fit_transform(reviews)
        
        if X.shape[0] != len(labels):
            raise ValueError("Number of samples in features and labels must be the same.")
        
        self.clf = MultinomialNB()
        self.clf.fit(X, labels)
        return make_pipeline(vectorizer, self.clf)

    def analyze_sentiment(self, review):
        sentiment_score = self.sia.polarity_scores(str(review))["compound"]
        return sentiment_score

# Update Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    # Columns to analyze
    feedback_columns = ['teaching', 'library_facilities', 'examination', 'labwork', 'extracurricular', 'coursecontent']
    sentiments = {}

    for column in feedback_columns:
        if column in df.columns:
            sentiments[column] = df[column].apply(analyzer.analyze_sentiment)

    overall_sentiments = {column: sum(sentiments[column]) / len(sentiments[column]) for column in feedback_columns}

    # Plotting sentiment analysis for all categories
    fig, ax = plt.subplots()
    categories = list(overall_sentiments.keys())
    sentiment_scores = list(overall_sentiments.values())

    ax.bar(categories, sentiment_scores, color=['blue', 'green', 'gray', 'red', 'purple', 'orange'])
    ax.set_xlabel('Feedback Categories')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Overall Sentiment Analysis for Feedback Categories')
    st.pyplot(fig)

    # Displaying descriptions
    st.subheader("Overall Sentiment Descriptions")
    for column in feedback_columns:
        avg_sentiment = sum(sentiments[column]) / len(sentiments[column])
        if avg_sentiment >= 0.65:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 0.62:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"
        st.write(f"**{column.capitalize()}**: {description}")

    # Train Naive Bayes classifier
    st.subheader("Naive Bayes Classifier")
    reviews = df[feedback_columns].values.flatten().tolist()
    labels = [1 if sentiment >= 0.65 else 0 for sublist in sentiments.values() for sentiment in sublist]
    pipeline = analyzer.train_classifier(reviews, labels)
    st.write("Classifier trained successfully.")

    # Prediction on new data
    test_reviews = st.text_area("Enter reviews for prediction (separate each review with a new line):")
    if test_reviews:
        test_reviews_list = test_reviews.split('\n')
        predictions = pipeline.predict(test_reviews_list)
        st.write("Predictions:")
        st.write(predictions)
else:
    st.write("Please upload a CSV file to proceed.")
