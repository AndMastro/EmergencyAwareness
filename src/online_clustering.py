import time
import sys
import numpy as np

from dateutil import parser

from FeatureExtractor import cbow_feature

from utils import create_windows, stemming_tokenizer, tokenize_tweet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import silhouette_score

'''
class to represent a cluster
'''
class Cluster:

    name_id = -1

    def __init__(self, is_tfidf):
        Cluster.name_id += 1
        
        self.is_tfidf = is_tfidf

        self.id = Cluster.name_id
        
        self.docs = []
        self.times = []

        self.time_centroid = 0
        self.centroid = np.array([0.0]*200)

    def add(self, tweet, tfidf_vec = None):
        tweet_text = tweet[1]
        tweet_time = time.mktime(tweet[0].timetuple())

        self.docs.append(tweet_text)

        n = len(self.docs)
        
        if self.is_tfidf:
            if tfidf_vec == None:
                raise Exception
            else:
                feature_vec = (tfidf_vec.transform([tweet_text])).toarray()[0]
        else:
            feature_vec = np.array(cbow_feature(tweet_text, average = True))

        if n == 1:
            self.centroid = feature_vec
        else:
            self.centroid = (self.centroid*(n-1)+feature_vec)/n

        if len(self.times) == 0:
            tweet_time -= 1 # we make this to avoid 0 variance 
        
        self.times.append(tweet_time)
        self.last_edit = tweet_time 

        self.time_centroid = (self.time_centroid*(n-1)+tweet_time )/n


    def update_centroid(self, tfidf_vec):
        if self.is_tfidf:
            vecs = list()
            for tweet in self.docs:
                #tokenize tweet
                feature_vec = (tfidf_vec.transform([tweet])).toarray()[0]
                vecs.append(feature_vec)        
            self.centroid = np.average(vecs, axis = 0)
        else:
            raise Exception

    def __str__(self):
        return str(self.id)
        
    def __repr__(self):
        return "Cluster(" + str(self.id) + ")"

    def __eq__(self, other):
        return self.id == other.id

'''
Clusterize a tweet in one of the active clusters, tweet is passed as a pair (date, text)
We can pass a similarity function of sklear, like cosine_similarity, or jaccard_similarity_score
'''
def clusterize_tweet(tweet, active_clusters, delta, similarity, use_tfidf = False, vectorizer = None):
    if use_tfidf:
        if vectorizer == None:
            raise Exception
        else:
            feature_vector = (vectorizer.transform([tweet[1]])).toarray()[0]
    else:
        feature_vector = np.array(cbow_feature(tweet[1], average = True))

    tweet_time = time.mktime(tweet[0].timetuple())
    max_sim = -1

    for c in active_clusters:
        feature_centroid = c.centroid
        time_centroid = c.time_centroid
        sigma2 = np.var(c.times + [tweet_time])
        mult = np.exp(-(((time_centroid-tweet_time)*(time_centroid-tweet_time)/(2*sigma2))))
        sim = similarity([feature_centroid], [feature_vector]) #higher is better
        sim = sim*mult
        
        if sim > delta and sim > max_sim:
            max_sim = sim
            add_to = c
    
    if max_sim > -1:
        add_to.add(tweet, vectorizer)
    else: 
        c = Cluster(use_tfidf)
        c.add(tweet, vectorizer)
        active_clusters.append(c)
    return c.id

'''
wraps function to create clusters
'''
def clustering_on_window(time_window, active_clusters, inactive_clusters, delta, time_delta, similarity, use_tfidf = False, vectorizer = None):
    tweets = [(parser.parse(tweet['created_at']), tweet['full_text']) for tweet in time_window]
    
    if use_tfidf:
        #recompute tf-idf matrix for the window
        docs_to_fit = [tweet[1] for tweet in tweets]
        for c in active_clusters:
            docs_to_fit += c.docs
    
        vectorizer.fit(docs_to_fit)

        for c in active_clusters:
            c.update_centroid(vectorizer)

    for tweet in tweets:
        clusterize_tweet(tweet, active_clusters, delta, similarity, use_tfidf, vectorizer)
        #sys.stdout.write("\rActive Clusters: %d" % len(active_clusters))

    last_time = time.mktime(tweet[0].timetuple())
    for cl in active_clusters:
        last_edit = cl.last_edit
        if (last_time - last_edit > time_delta):
            active_clusters.remove(cl)
            inactive_clusters.append(cl)

'''
'''
def print_silhouette(active_cluster, inactive_cluster, use_tfidf = False , vectorizer = None):
    tweets = []
    labels = []
    for c in active_cluster:
        docs = c.docs
        for doc in docs:
            tweets.append(doc)
            labels.append(c.id)
    for c in inactive_cluster:
        docs = c.docs
        for doc in docs:
            tweets.append(doc)
            labels.append(c.id)

    if use_tfidf:
        t_list = vectorizer.fit_transform(tweets)
    else:
        t_list = [cbow_feature(tweet, average = True) for tweet in tweets]
    
    score = silhouette_score(t_list, labels)
    return score

'''
To change pls, it's awful
'''
def jaccard_similarity(list1, list2):
    num = 0.0
    den = 0.0
    n = len(list1)
    i = 0
    while i < n:
        num += min(list1[i],list2[i])
        den += max(list1[i],list2[i])
        i += 1
    return (num / den)

#fix this
def final_wrapper(file_name_ds, use_tfidf):
    windows = create_windows(file_name_ds)
    
    vec = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize_tweet)
    similarity = cosine_similarity
    #similarity = jaccard_similarity
    
    delta = 0.05
    time_delta = 3600
    
    active_clusters = []
    inactive_clusters = []
    
    i = 0
    for window in windows:
        print("------\nWindow No: ", i)
        print("Size: ", len(window))
        clustering_on_window(window, active_clusters, inactive_clusters, delta, time_delta, similarity, use_tfidf, vec)
        print("Active Clusters: ", len(active_clusters))
        print("Inactive Clusters: ", len(inactive_clusters))
        i += 1

    print(print_silhouette(active_clusters, inactive_clusters, use_tfidf, vec))

#Main
#dataset_name = 'datasets/dataset/2012-10-29.txt'
#final_wrapper(dataset_name, False)