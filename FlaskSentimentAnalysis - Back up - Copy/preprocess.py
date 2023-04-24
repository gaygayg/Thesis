import pandas as pd
import numpy as np
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stopwordsiso import stopwords as tagalog_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# Load the dataset
#df = pd.read_csv('filtered_data5.csv', encoding='ISO-8859-1')

# Define a function to preprocess the tweet/text
def preprocess_text(text,processed_tweets):
    if isinstance(text, float):
        return ""
    # Remove URLs, mentions, and non-alphanumeric characters
    text = re.sub(r'http\S+|www.\S+|@\S+|[^\w\s]', '', text.lower())
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()
    # Split the text into words
    words = text.split()
    # Remove english and tagalog stop words and lemmatize the words
    lemmatizer = WordNetLemmatizer()
    english_stop_words = set(stopwords.words('english'))
    tagalog_stop_words = set(tagalog_stopwords(['tl']))
    stop_words = english_stop_words.union(tagalog_stop_words)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join the words back into a string
    processed_text = " ".join(words)

    # Check if the processed text is a duplicate
    if processed_text in processed_tweets:
        return None
    else:
        processed_tweets.add(processed_text)
        return processed_text
# Preprocess the text in the DataFrame
#df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Remove duplicate tweets and keep only the first occurrence
#df.drop_duplicates(subset='Tweet', keep='first', inplace=True)

# Save the preprocessed DataFrame back to the same CSV file
#df.to_csv('filtered_data05.csv', index=False)

# Show the preprocessed dataset
#print(df['Tweet'])