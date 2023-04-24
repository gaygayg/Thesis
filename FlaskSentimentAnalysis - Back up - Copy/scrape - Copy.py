import tweepy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
import requests
import json
from filter import filter_ads, unwanted_keywords
# Authenticate your Twitter API credentials

bearer_token = 'AAAAAAAAAAAAAAAAAAAAABl7kgEAAAAA40ilCTfClozo5%2B68tKnf1%2BVASwk%3D4pBD9ILkJnPFXC1l4ubWSPzjhL7ie1eXRwoS5NcgO14bQyzIv4'

lang = ["en", "tl"]

query = '#gcash'
max_results = 10

headers = {
    'Authorization': f'Bearer {bearer_token}'
}

params = {
    'query': query,
    'max_results': max_results,
    'tweet.fields': 'created_at,lang',

}

url = 'https://api.twitter.com/2/tweets/search/recent'

response = requests.get(url, headers=headers, params=params)

if response.status_code != 200:
    raise Exception(f'Request failed with error {response.status_code}: {response.text}')

tweets = response.json()['data']
filtered_tweets = []

for tweet in tweets:
    if len(filtered_tweets) == max_results:
        break
        
    # Check if the tweet has the expected keys
    if 'date' in tweet.keys() and 'text' in tweet.keys():
        tweet_date = tweet['date']
        tweet_content = tweet['text']
        
        filtered_tweets.append([tweet_date, tweet_content])
    else:
        print('Skipping tweet with unexpected structure:', tweet)
    
# Filter out rows containing unwanted keywords or phrases from the processed tweets
filtered_tweets = filter_ads(filtered_tweets)

# Create a Pandas DataFrame from the list of tweets
df = pd.DataFrame(filtered_tweets, columns=['Date', 'Tweet'])

print(df)

# Save the preprocessed DataFrame back to the same CSV file
df.to_csv('scrapedbearer.csv', index=False)


-------------------------------------------------------------import os
from flask import Flask, render_template, request, redirect
import snscrape.modules.twitter as sntwitter
import pandas as pd
import nltk
from datetime import datetime, timedelta
import re
import string
import tweepy
import pickle
import urllib.parse
# Import the necessary packages
from requests.structures import CaseInsensitiveDict
#import remove irrelevant tweets from filter.py
from filter import filter_ads, unwanted_keywords
# Import the preprocess function from preproces.py
from preprocess import preprocess_text
# Import the generate_report function from generate_report.py
from generate_report import generate_report
''''
@app.route('/download-csv')
def download_csv():
    data = [['Tweets', 'Positive', 'Negative'],
            [data_overview['tweets'], data_overview['positive'], data_overview['negative']]]

    # Create a response object with the CSV file
    response = make_response(convert_to_csv(data))
    response.headers['Content-Disposition'] = 'attachment; filename=twitter_search_results.csv'
    response.headers['Content-Type'] = 'text/csv'

    return response

# Helper function to convert a list of lists to a CSV file
def convert_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    for row in data:
        writer.writerow(row)
    return output.getvalue()
'''


# Set up the authentication
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABl7kgEAAAAA40ilCTfClozo5%2B68tKnf1%2BVASwk%3D4pBD9ILkJnPFXC1l4ubWSPzjhL7ie1eXRwoS5NcgO14bQyzIv4"
headers = CaseInsensitiveDict()
headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
headers["User-Agent"] = "GLEOMAY"
# Define the start and end date for the search and set language to enlish and tagalog
today = datetime.today()
start_date = today - timedelta(days=1)
end_date = today
languages = ["en", "tl"]
with open('mnb_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
'''
import os
from flask import Flask, render_template, request, redirect, response
import snscrape.modules.twitter as sntwitter
import pandas as pd
import nltk
from datetime import datetime, timedelta
import re
import string
import tweepy
import pickle
import urllib.parse
# Import the necessary packages
from requests.structures import CaseInsensitiveDict
#import remove irrelevant tweets from filter.py
from filter import filter_ads, unwanted_keywords
# Import the preprocess function from preproces.py
from preprocess import preprocess_text
# Import the generate_report function from generate_report.py
from generate_report import generate_report

# Set up the authentication
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABl7kgEAAAAA40ilCTfClozo5%2B68tKnf1%2BVASwk%3D4pBD9ILkJnPFXC1l4ubWSPzjhL7ie1eXRwoS5NcgO14bQyzIv4"
headers = CaseInsensitiveDict()
headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
headers["User-Agent"] = "GLEOMAY"
# Define the start and end date for the search and set language to enlish and tagalog
today = datetime.today()
start_date = today - timedelta(days=1)
end_date = today
languages = ["en", "tl"]
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
    if len(query.split()) == 1 and re.match(r'^#[A-Za-z]{3,}$', query):
        keyword = query
    else:
        return render_template('index.html', message='Please enter a single valid hashtagged keyword.', submitted=submitted)

    # Use snscrape to gather tweets based on the user's query
    tweets = []
    processed_tweets = set()
    for tweet in sntwitter.TwitterSearchScraper(keyword).get_items():
        if len(tweets) == 100:
            break
        elif tweet.lang in languages:
            # Check if the tweet content contains any unwanted keywords or phrases
            if any(keyword in tweet.content for keyword in unwanted_keywords):
                continue
            else:
                # Preprocess the tweet content using the preprocess function
                preprocessed_tweet = preprocess_text(tweet.content, processed_tweets)
                if preprocessed_tweet:
                    tweets.append([tweet.date, preprocessed_tweet, tweet.content])
    # Check if any tweets were found
    if not tweets:
        return render_template('index.html', message='Sorry, no tweets were found for this keyword.', submitted=submitted)

    # Filter out rows containing unwanted keywords or phrases from the processed tweets
    tweets = filter_ads(tweets)
    vectorized_tweet = vectorizer.transform([preprocessed_tweet])
    sentiment_score = classifier.predict(vectorized_tweet)[0]

    # Create a Pandas data frame from the list of tweets
    df = pd.DataFrame(tweets, columns=['Date', 'Tweet','Raw Tweet'])
 
    df['Sentiment Score'] =  sentiment_score

    # Remove rows with None values in the preprocessed text column
    df.dropna(subset=['Tweet'], inplace=True)

    # Convert the data frame to HTML, including only the desired columns
    data_table = df[['Date', 'Raw Tweet', 'Sentiment Score']].to_html(index=False)

    # Call the generate_report function with the data frame
    wc_image_url, sentiment_trend_image_url, accuracy, f1score, recall, precision, num_tweets,num_positive_tweets,num_negative_tweets = generate_report(df)

    # Render the template with the data frame and generated data
    return render_template('result.html', data=data_table, wc_image_url=wc_image_url, sentiment_trend_image_url=sentiment_trend_image_url, accuracy=accuracy, f1score=f1score, recall=recall, precision=precision, num_tweets=num_tweets,num_positive_tweets=num_positive_tweets,num_negative_tweets=num_negative_tweets)

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

    '''