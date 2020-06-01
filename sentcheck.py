import nltk
from tkinter import *
import tweepy
#nltk.download('punkt')
#nltk.download('twitter_samples')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from twython import Twython

import pandas as pd
import json
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random


top= Tk()

top.title('Sentiment Analysis')
top.geometry('500x500')

with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

# Instantiate an object
python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

auth = tweepy.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
api = tweepy.API(auth)


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    all_neg_words = get_all_words(negative_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    freq_dist_neg = FreqDist(all_neg_words)
    print('\n \n Top 10 most common positve words ')
    print(freq_dist_pos.most_common(10))
    print('\n \n Top 10 most common negative words ')
    print(freq_dist_neg.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("\n Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    
    

    L1 = Label(top, text="Enter Phrase/Hashtag for Analysis")
    L1.grid(row= 1, column= 2)
    E1= Entry(top, bd=5)
    E1.grid(row=5, column= 2)
    v= StringVar()
    v1=StringVar()
    v2=StringVar()
    L2 = Label(top, text="Answer", textvariable=v)
    L2.grid(row= 13, column= 5)

    l3=Label(top,text="Answer",textvariable=v1)
    l3.grid(row= 15, column=5)
    l4=Label(top,text="Answer",textvariable=v2)
    l4.grid(row=17,column=5)
    

    def checktweet():
        v.set(".")
        v1.set(".")
        v2.set(".")
        po=0
        ne=0
        que=E1.get()

        dict_ = {'text': []}
        for tweet in tweepy.Cursor(api.search, q=que,lang='en', rpp=100).items(50):
            custom_tokens = remove_noise(word_tokenize(tweet.text))
            ans= classifier.classify(dict([token, True] for token in custom_tokens))
            if ans == 'Positive':
                po=po+1
                print('Positive')
            elif ans == 'Negative':
                ne=ne+1
                print('Negative')
        if(po>0 or ne>0):
            t= 'Positive Tweets ' + str((po/(ne+po)))
            v.set(t)
        else:
            v.set('No Tweets Retrieved')
        
        t1='Negative Tweets  : '+str((ne/()ne+po))
        v1.set(t1)
        t2='Total tweets : '+str(po+ne)
        v2.set(t2)


    B = Button(top, text ="Check", command = checktweet)
    B.grid(row=9, column= 5)
    


    
    top.mainloop()
