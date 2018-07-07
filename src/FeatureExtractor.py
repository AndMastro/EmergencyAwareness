import string

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
import gensim

from operator import add

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVC

from tokenLib import *
import sys
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import numpy as np

import time

import pprint as pp
import json

def generate_buckets(number_of_buckets, start_date = "2012-10-23 00:00:00", end_date="2012-11-08 00:00:00"):
    start_time = time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S"))
    end_time = time.mktime(time.strptime(end_date, "%Y-%m-%d %H:%M:%S"))
    buckets = [start_time]

    interval_size = (end_time - start_time) / number_of_buckets

    for i in range(number_of_buckets):
        start_time += interval_size
        buckets.append(start_time)

    return buckets


def time_bucket(date, startDate = "2012-10-23 00:00:00", endDate = "2012-11-08 00:00:00", number_of_buckets=0, one_hot=True):  
    buckets = None
    if not buckets:
        buckets = generate_buckets(number_of_buckets, start_date=startDate, end_date=endDate)
    if not date:
        if one_hot:
            return [0]*len(buckets)
        else:
            return [0]
    try:            
        date_time = time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S"))

        for i in range(len(buckets)):
            if date_time >= buckets[i] and date_time < buckets[i+1]:
                if one_hot:
                    result = [0]*len(buckets)
                    result[i] = 1
                    return result
                else:
                    return [i+1]

        if date_time < buckets[0]:
            return [1] + [0]*(len(buckets)-1)
        elif date_time >= buckets[len(buckets)-1]:
            return [0] * (len(buckets)-1) + [1]
        else:
            print ("Bucketing failed. Datetime : " + str(date_time))
            sys.exit(2)
    except Exception as e:
        print ("Bucketing failed...!")
        print ("Exception : " + str(e))
        print ("Attempted date_time : " + str(date_time))
        print ("Returning zero vector")
        if one_hot:
            return [0]*len(buckets)
        else:
            return [0]

#choose between Standford model and our model
W2V_MODEL = "models/glove.6B.200d.txt"
#W2V_MODEL = "models/sockModel.txt"
w2v=None
def cbow_feature(tweet, average=False):
    global w2v
    if not w2v:
        w2v = {}
        modelFile = open(W2V_MODEL, 'r', encoding = 'utf-8')
        lines = modelFile.readlines()
        for line in lines:
            words = line.strip('\n').split(" ")
            w2v[words[0]] = list(map(float, words[1:]))
        
    res = [0]*200
    
    count = 0

    for word in tweet:
        if word in w2v:
            count += 1
            res = list(map(add, res, w2v[word]))
    
    if average:
        if count != 0:
            result = list(map(lambda x: x/count, res))
        else:
            result = res
    else:
        result = res
            
    return result

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, index):
        self.index = index

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        features = []
        for tweet in data:
            features.append(tweet[self.index])
        return features

class OneHotDate(BaseEstimator, TransformerMixin):
    
    def __init__(self, buckNum):
        self.buckNum = buckNum

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        oneHotBuckets = []
        for tweetDate in data:
            #print(tweet[2])
            oneHotBuckets.append(time_bucket(tweetDate, number_of_buckets = self.buckNum))
        return oneHotBuckets

class WordEmbeddings(BaseEstimator, TransformerMixin):
    
    def __init__(self, avg):
        self.avg = avg

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        wordEmbeddings = []
        for tweetText in data:
            wordEmbeddings.append(cbow_feature(tweetText, average = self.avg))
        return wordEmbeddings

class GetContext(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        context = []
        found = 0
        prev1 = ""
        prev2 = ""
        for tweet in data:
            for t in data:
                if tweet[3] in t: #it is the prev tweet ID
                    prev1 = t[1]
                    found += 1
                    for t2 in data:
                        if t[1] in t2:
                            prev2 = t2[1]
                            found += 2
            if found == 0:
                context.append(tweet[1])
            elif found == 1:
                context.append(prev1)
            else:
                context.append(prev1 + " " + prev2)
                
            found = 0
            
            
        return context

