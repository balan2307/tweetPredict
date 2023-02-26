from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import neattext.functions as nfx
import urllib.parse
import requests
import os
import json
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import preprocessing
from neattext.functions import clean_text
import numpy as np
import re
import warnings
import statistics
from flask import Flask, render_template, url_for, request
import pymongo
warnings.filterwarnings('ignore')
from pymongo import MongoClient
from bson.json_util import dumps
# Download stopwords
nltk.download('stopwords')

df = pd.read_csv('/app/data/tweet.csv')

# 1: convert the type column to int values
df.loc[df["Type"] == "medical-emergency", "Type"] = 1
df.loc[df["Type"] == "robbery", "Type"] = 2
df.loc[df["Type"] == "legal-emergency", "Type"] = 3
df.loc[df["Type"] == "supply-emergency", "Type"] = 4
df.loc[df["Type"] == "feedback", "Type"] = 5

# 2: clean the data
df['Tweet'] = df['Tweet'].apply(nfx.remove_multiple_spaces)
df['Tweet'] = df['Tweet'].apply(nfx.remove_urls)
df['Tweet'] = df['Tweet'].apply(nfx.remove_puncts)

# 3: Initializing TF-IDF vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(
    use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

# 4: convert the tweet to float values using TF-IDF vectorizer and initialize x and y
x = vectorizer.fit_transform(df.Tweet)
df.Type = df.Type.astype('int')
y = df.Type

# 5: Use the Random Forest Classifier
rf = RandomForestClassifier()

# 6: Initialize stratified k fold for training the model
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# 7: fitting the model
lst_accu_stratified = []

for train_index, test_index in skf.split(x, y):
    x_train_fold, x_test_fold = x[train_index], x[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    rf.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(rf.score(x_test_fold, y_test_fold))

# 8: sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0:
        sentiment_label = 'Positive'
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    result = {'polarity': sentiment_polarity,
              'subjectivity': sentiment_subjectivity,
              'sentiment': sentiment_label}
    ans = sentiment_label
    return ans

# 9: Database
client = MongoClient("mongodb+srv://Rutujeet:Rutujeet28@cluster0.u99f7ke.mongodb.net/?retryWrites=true&w=majority")
db = client.get_database('railway_db')
records = db.tweet

output_record = db.sample

all_records = records.find()

# 10: Final Prediction
def get_stream(set):
    stream = client.get_database('railway_db')
    change_stream = stream.test.watch()
    for change in change_stream:
        string = dumps(change["fullDocument"]["tweet"])
        print(string)
        inp = string
        docx = re.sub(r'#', '', inp)
        docx = re.sub(r'@[A-Za-z0-9]+', '', docx)
        docx = re.sub(r'RT[\s]+', '', docx)
        docx = re.sub(r'https?:\/\/\S+', '', docx)
        docx = re.sub(r'"\[A-Za-z0-9]"+', '', docx)
        docx = re.sub(r'\[n]+', '', docx)
        docx = clean_text(inp, urls=True, puncts=True)
        # , stopwords=True
        emergency = np.array([docx])
        emergency_vector = vectorizer.transform(emergency)
        answer = rf.predict(emergency_vector)
        print(inp)
        ans = ""
        result = ""
        value = dumps(change["fullDocument"]["tweet_ID"])
        if (answer == 1):
            result = "Medical-emergency"
            ans = "Emergency"
        elif (answer == 2):
            result = "robbery"
            ans = "Emergency"
        elif (answer == 3):
            result = "legal-emergency"
            ans = "Emergency"
        elif (answer == 4):
            result = "supply-emergency"
            ans = "Emergency"
        else:
            val = get_sentiment(docx)
            if (val == "Positive"):
                print("We are happy you enjoyed the journey")
                ans = val
            elif (val == "Negative"):
                print("Sorry for the inconvinience")
                ans = val
            else:
                ans = "Neutral"
                result = "feedback"

        print(result)
        value_of_ranking = answer.item()
        print(type(value_of_ranking))
        new_data = {
            "tweet_ID" : value,
            "tweet" : string,
            "prediction" : result,
            "Type" : ans,
            "ranking" :  value_of_ranking, 
            "responded" : False
        }
        output_record.insert_one(new_data)
        print(ans)

def main():
    get_stream(set)

if __name__ == "__main__":
    main()
