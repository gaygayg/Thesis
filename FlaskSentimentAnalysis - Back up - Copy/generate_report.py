import pandas as pd
from io import BytesIO
import base64
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

with open('mnb_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


def generate_report(df):
        # Data Overview
    num_tweets = len(df)
    num_positive_tweets = len(df[df['Sentiment Score'] == 'positive'])
    num_negative_tweets = len(df[df['Sentiment Score'] == 'negative'])

    #data_overview = f"Tweets {num_tweets} Positive {num_positive_tweets} Negative {num_negative_tweets}"

    # Generate performance metrics
    y_true = df['Sentiment Score']
    y_pred = classifier.predict(vectorizer.transform(df['Tweet']))
    accuracy = "{:.1%}".format(metrics.accuracy_score(y_true, y_pred))
    precision = "{:.1%}".format(metrics.precision_score(y_true, y_pred, pos_label="positive"))
    recall = "{:.1%}".format(metrics.recall_score(y_true, y_pred, average="weighted"))
    f1score = "{:.1%}".format(metrics.f1_score(y_true, y_pred, pos_label="positive"))
    
   # Generate wordcloud
    # Load Twitter icon mask and convert to RGB
    twitter_mask = np.array(Image.open('twitter_icon.png'))
    
    # Combine English and Tagalog stopwords
    stopwords_list = stopwords.words('english') + stopwords.words('tagalog')

    # Remove stopwords from the text data
    text = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords_list]))
    
    wc = WordCloud(background_color='white', mask=twitter_mask).generate(' '.join(text))
    
    # Set color of wordcloud to match Twitter logo
    image_colors = ImageColorGenerator(twitter_mask)
    wc.recolor(color_func=image_colors)
    
    # Save wordcloud as image
    wc_image = BytesIO()
    plt.figure(figsize=None)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(wc_image, format='png')
    wc_image.seek(0)
    wc_image_url = base64.b64encode(wc_image.getvalue()).decode()

    # Generate sentiment trend graph
    df['Datetime'] = pd.to_datetime(df['Date'])
    df.set_index('Datetime', inplace=True)
    df_resampled = df.resample('D').count() # Resample the dataset to include all dates
    df_pos = df[df['Sentiment Score'] == 'positive'].resample('D').count()
    df_neg = df[df['Sentiment Score'] == 'negative'].resample('D').count()
    plt.figure(figsize=(None))
    # Plot positive sentiment trend with dots on edges
    plt.plot(df_pos.index, df_pos['Tweet'], 'g-', label='Positive', marker='o', markerfacecolor='g', markersize=5)
    # Plot negative sentiment trend with dots on edges
    plt.plot(df_neg.index, df_neg['Tweet'], 'r-', label='Negative', marker='o', markerfacecolor='r', markersize=5)
    # Format x-axis ticks as "Mon. DD"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b. %d'))
    plt.gcf().autofmt_xdate(rotation=25) # Rotate x-axis labels for better readability
    plt.legend()
    plt.ylabel('Number of Tweets')
    plt.title('Sentiment Trend by Date')
    sentiment_trend_image = BytesIO()
    plt.savefig(sentiment_trend_image, format='png')
    sentiment_trend_image.seek(0)
    sentiment_trend_image_url = base64.b64encode(sentiment_trend_image.getvalue()).decode()

    return wc_image_url, sentiment_trend_image_url, accuracy, f1score, recall, precision, num_tweets,num_positive_tweets,num_negative_tweets