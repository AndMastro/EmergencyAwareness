import math
import json

import numpy as np

from os.path import join
from dateutil import parser
from datetime import timedelta

from scipy.stats import binom
from utils import tokenize_tweet, bl_rt, dump_json_to_file, get_files_in_dir, contain_a_word, create_windows

'''
Get the list of tweet id suspected of containing bursty features, we may want to print ome graph for the features detected
'''
def get_bursty_tweet(time_window_tweets, expectation_features, train = True, print_stats = False):
    
    N = len(time_window_tweets)
    
    id_ret = set()
    
    # Tokenize all the tweets - reduce them to unigram features
    tokenized_tweets = [(tweet['id'],tokenize_tweet(tweet['full_text'])) for tweet in time_window_tweets]
    
    # Get a list feature - list of tweet_ids where the feature appear in
    bag_of_feature = {}
    for id_flist in tokenized_tweets:
        for feature in id_flist[1]:
            bag_of_feature[feature] = bag_of_feature.get(feature,set())
            bag_of_feature[feature].add(id_flist[0])
 
    #print(bag_of_feature.keys())
    for feature in bag_of_feature:
        #  Count feature, number of tweets it appear in and reporpotionate it  
        n_feature_appear_in = len(bag_of_feature[feature])
        
        #Probability f_j appears in the time windows P_o(n_{i,j})
        Prob_f_window = n_feature_appear_in/N
    
        feature_info = expectation_features.get(feature, [0,0])
        expected = feature_info[1]
        windows_feature_appeared = feature_info[0]

        ra = math.floor(expected*N) #max of the distribution
        rb = binom.interval(0.999,N,expected)[1] #point where distribution appraches 0 
        q = (rb + ra) / 2

        if (n_feature_appear_in >= q):
            id_ret = id_ret.union(bag_of_feature[feature])

        if train:
            expected = ((expected * windows_feature_appeared) + Prob_f_window)/(windows_feature_appeared+1)
            expectation_features[feature] = [windows_feature_appeared+1,expected]
 
        if print_stats:
            print("------------------------------------------------------------------------")
            print("Feature: ",feature)
            print("Window size: ", N)
            print("n_{i,j}: ", n_feature_appear_in)
            print("Probability in the time window of the feature: "+ str("{0:.2f}".format(Prob_f_window)))
            print("Expectation of the feature: "+ str("{0:.2f}".format(expected)))
            #plot_graph(N, expected,int(N*n_feature_appear_in/N_tweets_window), Prob_f_window)
            print("------------------------------------------------------------------------")
      
    return id_ret

'''
return TP, FP, TN and FN (in this order for the window and a list of feature to find)
Is it correct to do this?
'''
def stat_bursty_tweets(time_window_tweets, expectation_features, features_to_find):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    N = len(time_window_tweets)
    tokenized_tweets = [(tweet['id'],tokenize_tweet(tweet['full_text'])) for tweet in time_window_tweets]

    bag_of_feature = {}
    for id_flist in tokenized_tweets:
        for feature in id_flist[1]:
            bag_of_feature[feature] = bag_of_feature.get(feature,set())
            bag_of_feature[feature].add(id_flist[0])

    for feature in bag_of_feature:    
        
        #Determine whether te feature matches the truth or not
        is_bursty = bl_rt(feature, features_to_find)    
        
        n_feature_appear_in = len(bag_of_feature[feature])
        
        feature_info = expectation_features.get(feature, [0,0])
        expected = feature_info[1]
        
        ra = math.floor(expected*N)                     #max of the distribution
        rb = binom.interval(0.999,N,expected)[1]        #point where is (almost) 0 
        q = (rb + ra) / 2 
        
        if (n_feature_appear_in >= q):
            if is_bursty:
                TP += 1
            else:
                FP += 1
        else:
            if is_bursty:
                FN += 1
            else:
                TN += 1
    
    return TP, FP, TN, FN

'''
return false alarm rate and detection rate for the current collection of tweets
'''
def evaluate_window(time_window_tweets, expectation_features, features_to_find):
    
    TP, FP, TN, FN = stat_bursty_tweets(time_window_tweets, expectation_features, features_to_find)

    detection = TP/(TP+FN)*100
    flase_alarm = FP/(TN+FP)*100

    print("--------------")
    print("TP: ",TP)
    print("FP: ",FP)
    print("TN: ",TN)
    print("FN: ",FN)
    print('Detection Rate: ',"{0:.2f}".format(detection))
    print('False Alarm Rate: ',"{0:.2f}".format(flase_alarm))
    print("--------------")
    return detection, flase_alarm

'''
Execute the burst detection module on the specified file
'''
def run_burst_detection(file_name_ds, file_name_ex, window_size_minutes = 1, train = True):
    windows = create_windows(file_name_ds, window_size_minutes = window_size_minutes)

    print("Got ",len(windows)," time windows from ", file_name_ds)

    if len(windows) == 0:
        print("No tweets found, will be created an empty dataset")
        database = {}
        dump_json_to_file(file_name_ds, database)

    try:
        expectation_fd = open(file_name_ex, mode = 'r')
        expectation = json.load(expectation_fd)
        expectation_fd.close()
    except FileNotFoundError as FNE:
        print(str(FNE))
        print("No model found, creating an Empty one")
        expectation = {}
        dump_json_to_file(file_name_ex, expectation)    

    for window in windows:
        out_ids = get_bursty_tweet(window, expectation, train = train)
        print("Number of tweets in window: ",len(window))
        print("Number of bursty tweets: ",len(out_ids))
        if (train):
            dump_json_to_file(file_name_ex, expectation)


def get_and_evaluate(dataset_file, file_name_ex, bursty, window_size_minutes = 1, N = 100):
    f_ds = open(dataset_file, mode = 'r')
    tweets_dict = json.load(f_ds) 
    f_ds.close()

    tweets = {}
    i = 0

    for k in tweets_dict:
        tweet_text = tweets_dict[k]['full_text']
        if contain_a_word(tweet_text, bursty):
            tweets[k] = tweets_dict[k]
            i += 1
        elif i%100 == 0:
            tweets[k] = tweets_dict[k]
            i += 1
    
    f_ds = open('temp', mode = 'w')
    json.dump(tweets, f_ds)
    f_ds.close()

    windows = create_windows('temp', window_size_minutes = window_size_minutes)
    
    num_windows = len(windows)

    new_windows = []

    i = 0
    while (i < num_windows):
        new_window = []
        while (len(new_window) < N and i < num_windows):
            new_window += windows[i]
            i += 1
        new_windows.append(new_window)

    try:
        expectation_fd = open(file_name_ex, mode = 'r')
        expectation = json.load(expectation_fd)
        expectation_fd.close()
    except:
        return

    i = 0
    detection_t = 0.0
    false_alarm_t = 0.0

    print("Running test on ", len(new_windows), " time windows")

    for window in new_windows:
        detection, false_alarm = evaluate_window(window, expectation, bursty)
        detection_t += detection
        false_alarm_t += false_alarm
        i += 1
    print("Avg Detection: " + str("{0:.2f}".format(detection_t/i)))
    print("Avg False Alarm: " + str("{0:.2f}".format(false_alarm_t/i)))

#main of module
def evaluate(train = False):
    dataset_path = 'dataset/dataset'
    model_path = 'models/model_expectation_bd'
    test = '2012-10-29.txt'
    dataset_files = get_files_in_dir(dataset_path)
    dataset_files.remove(test)
    
    features_burst = ['Storm', 'Episode', 'Obama', 'Hurricane', 'Sandy', 'Game', 'Football', 'Giants', 'Cowboys', 'Romney', 'Debat', 'Frankenstorm', 'Halloween', 'TheWalkingDead', 'WalkingDead', 'Walking', 'Dead', '#Sandy', '#Hurricanesandy']

    if train:
        for file in dataset_files:
            print("Training on file: " + file + "...")
            run_burst_detection(join(dataset_path, file), model_path)

    get_and_evaluate(join(dataset_path, test), model_path, features_burst)
    
    #run_burst_detection(join(dataset_path, test), model_path,train = False)

#evaluate()