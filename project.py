import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
sentiment_classifier = pipeline("sentiment-analysis")

def main():
    st.title("Movie Review Sentiment Analysis")

    # Input text box for user to enter a movie review
    input_text = st.text_input('Enter a movie review:')

    if input_text:
        # Predict sentiment
        prediction = sentiment_classifier(input_text)

        # Extract sentiment label and score
        sentiment_label = prediction[0]['label']
        sentiment_score = prediction[0]['score']

        st.write(f'The sentiment of the review is: {sentiment_label} (Score: {sentiment_score:.2f})')

if __name__ == "__main__":
    main()
