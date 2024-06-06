
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

# Update Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()

    if 'teaching' in df.columns and 'coursecontent' in df.columns and 'examination' in df.columns and 'labwork' in df.columns and 'library_facilities' in df.columns and 'extracurricular' in df.columns:
            review_columns = df.columns[1::2]  # 
        reviews = df[review_columns].values.flatten().tolist()

        analyzer = SentimentAnalyzer()

        review_period = st.selectbox("Review Period:", [1, 4])

        if review_period == 1:
            sentiments = analyzer.analyze_sentiment(reviews)
        else:
            sentiments = analyzer.analyze_periodic_sentiment(reviews, review_period)

        overall_sentiment = analyzer.calculate_overall_sentiment(reviews)
        st.subheader(f"Overall Sentiment: {overall_sentiment:.2f}")
        st.subheader("Sentiment Analysis")
    
            # Initialize lists to store sentiment scores and labels
            all_reviews = []
    
            # Analyze sentiment for each aspect
            aspect_columns = ['teaching', 'coursecontent', 'examination', 'labwork', 'library_facilities', 'extracurricular']
            for column in aspect_columns:
                aspect_reviews = df[column].astype(str).tolist()
                all_reviews.extend(aspect_reviews)
    
            # Visualization
            for column in aspect_columns:
                analyzed_sentiments = analyzer.analyze_sentiment(df[column].astype(str).tolist())
                sentiment_scores = [sentiment['compound'] for sentiment in analyzed_sentiments]
                weeks = range(1, len(sentiment_scores) + 1)
                plt.plot(weeks, sentiment_scores, label=column.capitalize())
    
            plt.xlabel('Week')
            plt.ylabel('Sentiment Score')
            plt.title('Sentiment Trends for Each Aspect Over Weeks')
            plt.legend()
            st.pyplot()

    elif len(df.columns) >= 7 and df.columns[0].lower() == 'student' and all(col.lower().startswith('week') for col in df.columns[1:]):
        # Data structure suggests weekly sentiment analysis
        # Initialize lists to store sentiment scores and labels
        all_reviews = []
        sentiment_labels = []
    
        # Analyze sentiment for each week
        weekly_sentiments = {}
        for column in df.columns[1:]:
            weekly_reviews = df[column].dropna().astype(str).tolist()
            all_reviews.extend(weekly_reviews)
            analyzed_sentiments = analyzer.analyze_sentiment(weekly_reviews)
    
            # Store weekly sentiments for visualization
            weekly_sentiments[column] = analyzed_sentiments
    
            # Extract compound scores and determine sentiment labels (binary classification)
            compound_scores = [sentiment['compound'] for sentiment in analyzed_sentiments]
            weekly_labels = [1 if score > 5 else 0 for score in compound_scores]
            sentiment_labels.extend(weekly_labels)
    
        # Plotting sentiment trends
        weeks = list(range(1, len(df.columns)))
        sentiment_scores = [sum([sentiment['compound'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
        pos_scores = [sum([sentiment['pos'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
        neu_scores = [sum([sentiment['neu'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
        neg_scores = [sum([sentiment['neg'] for sentiment in weekly_sentiments[column]]) / len(weekly_sentiments[column]) for column in df.columns[1:]]
    
        fig, ax = plt.subplots()
        ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
        ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
        ax.plot(weeks, pos_scores, label="Positive", color="green")
        ax.plot(weeks, neu_scores, label="Neutral", color="gray")
        ax.plot(weeks, neg_scores, label="Negative", color="red")
    
        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Trend Over Weeks')
        ax.legend()
        st.pyplot(fig)
    
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
    
        # Individual student analysis
        st.subheader("Individual Student Analysis")
        for student in df.columns[1:]:
            st.write(f"**Student:** {student}")
            student_reviews = df[student].dropna().astype(str).tolist()
            student_sentiments = analyzer.analyze_sentiment(student_reviews)
            student_description, student_trend = analyzer.interpret_sentiment(student_sentiments)
            st.write(f"Sentiment Trend: {student_trend}")
            st.write(f"Description: {student_description}")

        
    else:
        st.write("Columns mismatch. Please ensure the CSV file contains the required columns.")

   
