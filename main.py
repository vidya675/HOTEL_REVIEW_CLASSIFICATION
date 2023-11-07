# Load libraries
import pandas as pd
import numpy as np
import warnings
from afinn import Afinn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string
import re
import pickle
import streamlit as st
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
ps = PorterStemmer()
# Function to clean and preprocess tweet text
stop_words = set(stopwords.words('english'))
def text_preprocessing(reviews):

    #convert to lower
    reviews=reviews.lower()

    #remove punctuation
    translator=str.maketrans('','',string.punctuation)
    reviews_without_punctuations=reviews.translate(translator)
    reviews_without_punctuations=re.sub(r'\d+','',reviews_without_punctuations)

    #tokenization
    tokens=word_tokenize(reviews_without_punctuations)

    #remove stop words
    stop_words=set(stopwords.words('english'))
    cleaned_tokens=[word for word in tokens if word not in stop_words]

    # Lemmitization
    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens=[lemmatizer.lemmatize(token) for token in cleaned_tokens]

    cleaned_reviews=' '.join(lemmatized_tokens).strip()

    return cleaned_reviews


def main():
    st.title('Welcome to Sentiment Analysis!')

    input_ = st.text_area('Enter the text')

    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    # Load final model
    model = pickle.load(open('Final_model.pkl', 'rb'))

    if st.button('Submit'):
        # Input from user
        cleaned_input = text_preprocessing(input_)
        # converting text to tfidf vectorizer
        array_input = tfidf.transform(pd.Series(cleaned_input)).toarray()
        # Prediction
        result = model.predict(array_input)
        # Result
        positive_emoji = "\U0001F604"  # Positive sentiment
        negative_emoji = "\U0001F61E"  # Negative sentiment
        neutral_emoji = "\U0001F610"   # Neutral sentiment
        if result == 0:
            st.header(f'Sentiment : Negative {negative_emoji}')
        elif result == 1:
            st.header(f'Sentiment : Neutral {neutral_emoji}')
        elif result == 2:
            st.header(f'Sentiment : Positive {positive_emoji}')

        afin = Afinn()
        word_list = cleaned_input.split(" ")
        pos_word = []
        neg_word = []
        neutral_word = []
        for i in word_list:
            score = afin.score(i)
            if score > 0:
                pos_word.append(i)
            elif score == 0:
                neutral_word.append(i)
            elif score < 0:
                neg_word.append(i)
        pos_colors = ["#4CAF50", "#ADD8E6", "#9370DB"]
        colored_pos_words = []
        st.subheader('Positive Keywords: ')
        # Display words row-wise with different text colors
        for i, word in enumerate(pos_word):
            # Use HTML to set the text color
            colored_word = f'<span style="color: {pos_colors[i % len(pos_colors)]};">{word}</span>'
            colored_pos_words.append(colored_word)

        # Join the colored words into a single string without line breaks
        words_row = ' '.join(colored_pos_words)
        st.markdown(words_row, unsafe_allow_html=True)

        neg_color = ["#FF0000", "#FFA07A", "#E6E6FA"]
        colored_neg_words = []
        st.subheader('Negative Keywords: ')
        for j, word in enumerate(neg_word):
            # Use HTML to set the text color
            colored_word = f'<span style="color: {neg_color[j % len(neg_color)]};">{word}</span>'
            colored_neg_words.append(colored_word)

        # Join the colored words into a single string without line breaks
        words_row = ' '.join(colored_neg_words)
        st.markdown(words_row, unsafe_allow_html=True)

        neutral_color = ["#A9A9A9", "#696969", "#BDB76B"]
        colored_neutral_words = []
        st.subheader('Neutral Keywords: ')
        for k, word in enumerate(neutral_word):
            # Use HTML to set the text color
            colored_word = f'<span style="color: {neutral_color[k % len(neutral_color)]};">{word}</span>'
            colored_neutral_words.append(colored_word)

        # Join the colored words into a single string without line breaks
        words_row = ' '.join(colored_neutral_words)
        st.markdown(words_row, unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()

