import streamlit as st
import pandas as pd
from collections import Counter
import math
import string

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/FDSA/DATASET/Movie_Review.csv')
    return df

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

def calculate_word_frequency(text):
    words = text.split()
    word_freq = Counter(words)
    return word_freq

def calculate_tf(word_freq):
    total_words = sum(word_freq.values())
    tf = {word: freq / total_words for word, freq in word_freq.items()}
    return tf

def calculate_idf(docs, word):
    num_docs_containing_word = sum(1 for doc in docs if word in doc)
    if num_docs_containing_word == 0:
        return 0
    else:
        return math.log(len(docs) / num_docs_containing_word)

def calculate_tfidf(tf, idf):
    return {word: tf * idf for word, tf in tf.items()}

def main():
    st.title("Movie Review Sentiment Analysis")
    
    df = load_data()

    # Input text box for user to enter a movie review
    input_text = st.text_input('Enter a movie review:')

    if input_text:
        input_text = preprocess_text(input_text)
        word_freq = calculate_word_frequency(input_text)
        tf = calculate_tf(word_freq)

        # Load all movie reviews
        docs = df['text'].apply(preprocess_text).tolist()
        idf = {word: calculate_idf(docs, word) for word in word_freq.keys()}

        # Calculate TF-IDF for the input text
        tfidf = calculate_tfidf(tf, idf)

        # Sentiment prediction based on TF-IDF
        positive_words = ['good', 'excellent', 'wonderful', 'great', 'amazing']
        negative_words = ['bad', 'poor', 'disappointing', 'terrible', 'awful']
        positive_score = sum(tfidf[word] for word in positive_words if word in tfidf)
        negative_score = sum(tfidf[word] for word in negative_words if word in tfidf)

        if positive_score > negative_score:
            sentiment = 'Positive'
        elif negative_score > positive_score:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        st.write(f'The sentiment of the review is: {sentiment}')

if __name__ == "__main__":
    main()
