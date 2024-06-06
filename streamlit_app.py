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

    # Extracting relevant columns for sentiment analysis
    relevant_columns = ['Teacher Feedback', 'Course Content', 'Examination pattern', 'Laboratory', 'Library Facilities', 'Extra Co-Curricular Activities', 'Any other suggestion']

    # Combine all feedback columns into one
    df['Combined Feedback'] = df[relevant_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Analyze sentiment for the combined feedback
    all_reviews = df['Combined Feedback'].tolist()
    overall_sentiments = analyzer.analyze_sentiment(all_reviews)

    # Interpret overall sentiment
    description, trend = analyzer.interpret_sentiment(overall_sentiments)

    st.subheader("Overall Analysis")
    st.write(f"Sentiment Trend: {trend}")
    st.write(f"Description: {description}")

    # Breakdown of analysis
    st.subheader("Breakdown of Analysis")
    breakdown_df = pd.DataFrame(overall_sentiments)
    st.write(breakdown_df)

    # Individual student analysis
    st.subheader("Individual Student Analysis")
    for index, row in df.iterrows():
        st.write(f"**Student:** {row['Name']}")
        student_reviews = [row[column] for column in relevant_columns]
        student_sentiments = analyzer.analyze_sentiment(student_reviews)
        student_description, student_trend = analyzer.interpret_sentiment(student_sentiments)
        st.write(f"Sentiment Trend: {student_trend}")
        st.write(f"Description: {student_description}")
