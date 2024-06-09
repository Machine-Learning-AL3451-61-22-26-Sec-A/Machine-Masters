import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/FDSA/DATASET/Movie_Review.csv')
    return df

def main():
    st.title("Movie Review Sentiment Analysis")
    
    df = load_data()

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    # Train the model
    model = MultinomialNB()
    model.fit(X, y)

    # Input text box for user to enter a movie review
    input_text = st.text_input('Enter a movie review:')

    if input_text:
        # Predict sentiment
        text_vec = vectorizer.transform([input_text])
        prediction = model.predict(text_vec)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        st.write(f'The sentiment of the review is: {sentiment}')

if __name__ == "__main__":
    main()
