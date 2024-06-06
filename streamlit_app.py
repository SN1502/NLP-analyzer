import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
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

    def calculate_overall_sentiment(self, reviews):
        compound_scores = [self.sia.polarity_scores(str(review))["compound"] for review in reviews if isinstance(review, str)]
        overall_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return self.transform_scale(overall_sentiment)

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(str(review))["compound"]),
                       'pos': self.sia.polarity_scores(str(review))["pos"],
                       'neu': self.sia.polarity_scores(str(review))["neu"],
                       'neg': self.sia.polarity_scores(str(review))["neg"]}
                      for review in reviews if isinstance(review, str)]
        return sentiments

    def analyze_periodic_sentiment(self, reviews, period):
        period_reviews = [' '.join(reviews[i:i + period]) for i in range(0, len(reviews), period)]
        return self.analyze_sentiment(period_reviews)

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

    # Assuming sentiment labels are spread across 'Week 1' to 'Week 6' columns
    sentiment_columns = df.columns[1:]  # Exclude the first column which contains student names

    # Concatenate sentiment data from all columns into a single list
    sentiments = df[sentiment_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()

    # Map sentiment labels to integers for training
    # You can adjust this mapping based on your sentiment classification needs
    sentiment_labels = {'positive': 1, 'neutral': 0, 'negative': -1}
    y = [sentiment_labels[sentiment.lower()] for sentiment in df['Sentiment']]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sentiments, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")
    
    # Visualization (if needed)
    # ...

    # Interpret sentiment analysis results
    description, trend = analyzer.interpret_sentiment(sentiments)
    st.subheader("Progress Description")
    st.write(f"Sentiment Trend: {trend}")
    st.write(f"Description: {description}")

    # Breakdown of analysis
    st.subheader("Breakdown of Analysis")
    breakdown_df = pd.DataFrame(sentiments, columns=sentiment_columns)
    st.write(breakdown_df)
