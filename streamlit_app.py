import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import io

# Downloading necessary NLTK data
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(str(review))["compound"]),
                       'pos': self.sia.polarity_scores(str(review))["pos"],
                       'neu': self.sia.polarity_scores(str(review))["neu"],
                       'neg': self.sia.polarity_scores(str(review))["neg"]}
                      for review in reviews if isinstance(review, str)]
        return sentiments

    def interpret_sentiment(self, sentiments):
        avg_sentiment = sum([sentiment['compound'] for sentiment in sentiments]) / len(sentiments) if sentiments else 0
        if avg_sentiment >= 6.5:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 6.2:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"

        trend = "No change"
        if len(sentiments) > 1:
            first_half_avg = sum([sentiment['compound'] for sentiment in sentiments[:len(sentiments)//2]]) / (len(sentiments)//2)
            second_half_avg = sum([sentiment['compound'] for sentiment in sentiments[len(sentiments)//2:]]) / (len(sentiments)//2)
            if second_half_avg > first_half_avg:
                trend = "Improving"
            elif second_half_avg < first_half_avg:
                trend = "Declining"

        return description, trend

# Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    # Assuming sentiment labels are spread across 'Teacher Feedback' to 'Any other suggestion' columns
    sentiment_columns = df.columns[5:]  # Exclude the first columns which are not review texts

    # Initialize lists to store sentiment scores and labels
    all_reviews = []
    sentiment_labels = []

    # Analyze sentiment for each column
    for column in sentiment_columns:
        column_reviews = df[column].dropna().astype(str).tolist()
        all_reviews.extend(column_reviews)
        analyzed_sentiments = analyzer.analyze_sentiment(column_reviews)

        # Extract compound scores and determine sentiment labels (binary classification)
        compound_scores = [sentiment['compound'] for sentiment in analyzed_sentiments]
        column_labels = [1 if score > 5 else 0 for score in compound_scores]
        sentiment_labels.extend(column_labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_reviews, sentiment_labels, test_size=0.2, random_state=42)

    # Convert text data to numeric using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_vectorized, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_vectorized)

    # Analyze all concatenated reviews for overall interpretation
    overall_sentiments = analyzer.analyze_sentiment(all_reviews)
    description, trend = analyzer.interpret_sentiment(overall_sentiments)

    st.subheader("Progress Description")
    st.write(f"Sentiment Trend: {trend}")
    st.write(f"Description: {description}")

    # Breakdown of analysis
    st.subheader("Breakdown of Analysis")
    breakdown_df = pd.DataFrame(overall_sentiments)
    st.write(breakdown_df)

    # Plotting sentiment trends
    if len(sentiment_columns) > 1:
        st.subheader("Sentiment Trends Over Columns")
        fig, ax = plt.subplots()
        for column in sentiment_columns:
            column_sentiments = [sentiment['compound'] for sentiment in analyzer.analyze_sentiment(df[column].dropna().astype(str).tolist())]
            ax.plot(range(len(column_sentiments)), column_sentiments, label=column)
        ax.set_xlabel('Review Index')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Trend Over Columns')
        ax.legend()
        st.pyplot(fig)
