import os
from flask import Flask, render_template, request, redirect, Response
import pandas as pd
import nltk
from datetime import datetime, timedelta
import re
import string
import pickle
# Import remove irrelevant tweets from filter.py
from filter import filter_ads, unwanted_keywords
# Import the preprocess function from preprocess.py
from preprocess import preprocess_text
# Import the generate_report function from generate_report.py
from generate_report import generate_report
from flask import make_response
from flask import send_file 
import io
import csv

from io import BytesIO


# Load the dataset
df = pd.read_csv('Datasets.csv', encoding='ISO-8859-1')

# Load the machine learning model and vectorizer
with open('mnb_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/search', methods=['POST'])
def search():
    # Get the user's query from the form
    query = request.form['query']

    # Check if the form has been submitted
    submitted = True if request.method == 'POST' else False

    # Check if the query is empty
    if not query:
        return render_template('index.html', message='Please enter a keyword before searching.', submitted=submitted)

    # Check if the query contains only one word and that word starts with a '#' character
    if len(query.split()) == 1 and re.match(r'^#[A-Za-z]{3,}$', query) and len(query) > 1:
        keyword = query
    else:
        return render_template('index.html', message='Please enter a single valid hashtagged keyword.', submitted=submitted)

    # Read the CSV file
    df = pd.read_csv('Datasets.csv')

    # Filter tweets containing the keyword
    keyword_df = df[df['Tweet'].str.contains(keyword, case=False)]

    # Check if the keyword is found in the CSV file
    if keyword_df.empty:
        return render_template('index.html', message=f'No tweets found for the keyword "{keyword}".', submitted=submitted)


    # Filter out rows containing unwanted keywords or phrases
    keyword_df = keyword_df[~keyword_df['Tweet'].str.contains('|'.join(unwanted_keywords), case=False)].head(100)

    # Preprocess the tweet content using the preprocess function
    keyword_df['Preprocessed Tweet'] = keyword_df['Tweet'].apply(lambda x: preprocess_text(x,processed_tweets=set()))

    # Vectorize the preprocessed tweets
    vectorized_tweets = vectorizer.transform(keyword_df['Preprocessed Tweet'])

    # Predict the sentiment score for each tweet
    sentiment_scores = classifier.predict(vectorized_tweets)

    # Add the sentiment scores to the data frame
    keyword_df['Sentiment Score'] = sentiment_scores

    
    # Select the columns to display in the HTML table
    columns = ['Date', 'Tweet', 'Sentiment Score']

    # Store the selected columns in a new data frame
    selected_df = keyword_df[columns]

    # Drop duplicates from the data frame
    selected_df.drop_duplicates(subset=['Tweet'], inplace=True)

    # Call the generate_report function with the data frame
    wc_image_url, sentiment_trend_image_url, accuracy, f1score, recall, precision, num_tweets,num_positive_tweets,num_negative_tweets = generate_report(selected_df)

    # Render the template with the data frame and generated data
    return render_template('result.html', data=selected_df.to_html(index=False), wc_image_url=wc_image_url, sentiment_trend_image_url=sentiment_trend_image_url, accuracy=accuracy, f1score=f1score, recall=recall, precision=precision, num_tweets=num_tweets,num_positive_tweets=num_positive_tweets,num_negative_tweets=num_negative_tweets)

@app.route('/exit')
def exit():
    return redirect('/')


@app.route('/download')
def download():
    # Get the data from the previous search
    data = request.args.get('data')

    # Convert the HTML table to a pandas dataframe
    df = pd.read_html(data)[0]

    # Create a CSV file from the dataframe
    csv_data = df.to_csv(index=False)

    # Send the file to the user for download
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=tweets.csv"})


if __name__ == '__main__':
    app.run(debug=True)