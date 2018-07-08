import json 
 
from burst_detection import get_bursty_tweet 
from utils import create_windows, dump_json_to_file 
from online_clustering import clustering_on_window, print_silhouette 
from sklearn.metrics.pairwise import cosine_similarity 

from sklearn.feature_extraction.text import TfidfVectorizer

from FeatureExtractor import cbow_feature
from utils import create_windows, tokenize_tweet


def run_clustering(day_name, delta, time_delta, use_burst = False, expectation_file = "", use_tfidf = False):

    if use_tfidf: 
        s1 = 'tfidf'
        vec = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize_tweet)  
    else: 
        s1 = 'we' 
        vec = None

    out = ""+s1+"_"+str(delta)+"_"+str(time_delta)+".csv" 

    expectation_fd = open(expectation_file, mode = 'r') 
    expectation = json.load(expectation_fd) 
    expectation_fd.close() 
    print("Expectation Model Loaded...") 

    database_fd = open(day_name, mode = 'r') 
    database = json.load(database_fd) 
    database_fd.close() 
    print("Database File Loaded...") 

    monthArray = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dic"] 
    try:
        for k in database: 
            database[k]['id_str'] = str(k) 
            database[k]['full_text'] = database[k]['text'] 
            date = database[k]['created_at'].split(" ") 
            if len(date) == 2: 
                date = date[0] + " " + date[1] 
            else: 
                date = str(date[5]) + "-" + str(monthArray.index(date[1])+1) + "-" +  str(date[2]) + " " + str(date[3]) 
            database[k]['created_at'] = date 
    except:
        print("Dataset Fixed")
    
    database_2 = database

    try:
        database_2 = {}     
 
        for k in database: 
            if database[k]['annotations'] != ['None']: 
                database_2[k] = database[k] 
    except:
        database_2 = database

    dump_json_to_file(out, database_2) 
    windows = create_windows(out) 

    active_clusters = [] 
    inactive_clusters = [] 

    i = 0 
 
    for window in windows:
        print("--------\nSize of Window: ",len(window)) 
        if use_burst:
            bursty_ids = get_bursty_tweet(window, expectation, train = False) 
            print("Bursty tweets Found: ", len(bursty_ids)) 
            
            bursty_tweets = [database[str(ids)] for ids in bursty_ids] 
            b_t = [(tweet['created_at'], tweet['id']) for tweet in bursty_tweets] 
            b_t.sort() 
            bursty_tweets = [database[str(tweet_tuple[1])] for tweet_tuple in b_t] 
        else:
            bursty_tweets = window
        print("Clustering...") 
        clustering_on_window(bursty_tweets, active_clusters, inactive_clusters, delta, time_delta, cosine_similarity, use_tfidf, vec) 
        print("Active clusters: " + str(len(active_clusters))) 
        print("Inctive clusters: " + str(len(inactive_clusters))) 
        i += 1 

    helper = {} 

    for el in database: 
        helper[database[el]['full_text']] = (el,database[el].get('annotations',[]))  
 
    out_fs = open(out, mode = 'w') 
 
    for c in active_clusters: 
        textes = c.docs 
        cluster_id = c.id 
        for text in textes: 
            info = helper[text] 
            print(info[0], '"' + text + '"', info[1], cluster_id, sep = ';',file = out_fs) 
    for c in inactive_clusters: 
        textes = c.docs 
        cluster_id = c.id 
        for text in textes: 
            info = helper[text] 
            print(info[0],  '"' + text + '"', info[1], cluster_id, sep = ';',file = out_fs) 
 
    out_fs.close() 

    ret = print_silhouette(active_clusters, inactive_clusters, use_tfidf, vec) 
    print(ret) 

day_name = 'dataset/dataset_merged.json' 
#day_name = 'dataset/dataset/2012-10-29.txt' 
expectation_file = 'models/model_expectation_bd'
delta = 0.05 
time_delta = 3600 
use_tfidf = False
use_burst = False

run_clustering(day_name, delta, time_delta, use_burst, expectation_file, use_tfidf)

