import json

import gc

from burst_detection import get_bursty_tweet
from utils import create_windows, dump_json_to_file
from online_clustering import clustering_on_window, print_silhouette

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import create_windows, stemming_tokenizer, tokenize_tweet

'''
day_name = 'dataset/dataset/2012-10-29.txt'
expectation_file = 'models/model_expectation_bd'

expectation_fd = open(expectation_file, mode = 'r')
expectation = json.load(expectation_fd)
expectation_fd.close()

print("Expectation Model Loaded...")

windows = create_windows(day_name)

print("Created Time windows...")

database_fd = open(day_name, mode = 'r')
database = json.load(database_fd)
database_fd.close()

print("Database File Loaded...")

active_clusters = []
inactive_clusters = []
delta = 0.05
time_delta = 3600

i = 0

for window in windows:
    print("--------\nSize of Window: ",len(window))
    bursty_ids = get_bursty_tweet(window, expectation, train = False)
    print("Bursty tweets Found: ", len(bursty_ids))

    bursty_tweets = [database[str(ids)] for ids in bursty_ids]
    b_t = [(tweet['created_at'], tweet['id']) for tweet in bursty_tweets]
    b_t.sort()
    bursty_tweets = [database[str(tweet_tuple[1])] for tweet_tuple in b_t]
    print("Clustering...")
    clustering_on_window(bursty_tweets, active_clusters, inactive_clusters, delta, time_delta, cosine_similarity, use_tfidf = False, vectorizer = None)
    print("Active clusters: " + str(len(active_clusters)))
    print("Inctive clusters: " + str(len(inactive_clusters)))
    i += 1

print("Speramo va")

gc.collect()

ret = print_silhouette(active_clusters, inactive_clusters, False, None)
print(ret)
'''

dataset_name = 'dataset/dataset_merged.json'
temp_dataset = 'tmp_rel.json' 
delta = 0.05
time_delta = 3600
use_tfidf = False

if use_tfidf:
    s1 = 'tfidf'
else:
    s1 = 'we'

out = "out/"+s1+"_"+str(delta)+"_"+str(time_delta)+"_rel.csv"

database_fd = open(dataset_name, mode = 'r')
database = json.load(database_fd)
database_fd.close()


monthArray = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dic"]
for k in database:
    database[k]['id_str'] = str(k)
    database[k]['full_text'] = database[k]['text']
    date = database[k]['created_at'].split(" ")
    if len(date) == 2:
        date = date[0] + " " + date[1]
    else:
        date = str(date[5]) + "-" + str(monthArray.index(date[1])+1) + "-" +  str(date[2]) + " " + str(date[3])
    database[k]['created_at'] = date

#
database_2 = {}    

for k in database:
    if database[k]['annotations'] != ['None']:
        database_2[k] = database[k]
#

dump_json_to_file(temp_dataset, database_2)

windows = create_windows(temp_dataset)
active_clusters = []
inactive_clusters = []

vec = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize_tweet)

for window in windows:
    clustering_on_window(window, active_clusters, inactive_clusters, delta, time_delta, cosine_similarity, use_tfidf = use_tfidf, vectorizer = vec)
    print("Active Clusters: ", len(active_clusters))
    print("Inactive Clusters: ", len(inactive_clusters))
helper = {}

for el in database:
   helper[database[el]['text']] = (el,database[el]['annotations']) 

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