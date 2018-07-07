import json
import string

from os.path import isfile, join
from os import listdir, makedirs, path

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer, EnglishStemmer

from scipy.stats import binom
import matplotlib.pyplot as plt

from dateutil import parser
from datetime import timedelta

'''
Dump a data structure as json into specified file
'''
def dump_json_to_file(f_name, structure):
    fd = open(f_name, mode = 'w')
    json.dump(structure, fd)
    fd.close()

'''
Get all the files in a directory as a list
'''
def get_files_in_dir(dir_name):
    return [file for file in listdir(dir_name) if isfile(join(dir_name, file))]

'''
Tokenize a tweet text, the function perform stemming, tokenization and removes stopwords
In addition, reduce all the hashtags (#) and mentions (@) to raw text 
'''
def tokenize_tweet(tweet_text, stemmer = SnowballStemmer('english'), tokenizer = TweetTokenizer(reduce_len=True), stopWords = set(stopwords.words('english'))):
    
    tokens = tokenizer.tokenize(tweet_text.lower())
    stemmed_tokens = []
    
    #stem the tokens
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))

    ret_tokens = []
    
    #remove stopwords
    for token in stemmed_tokens:
        if token not in stopWords and token not in string.punctuation and token.strip('@#').isalpha():
            ret_tokens.append(token)

    return ret_tokens


def stemming_tokenizer(text, stemmer = EnglishStemmer()):
	stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
	return stemmed_text

def tweet_tokenizer(text, stemmer = EnglishStemmer()):
	stemmed_text = [stemmer.stem(word) for word in TweetTokenizer(preserve_case = False, strip_handles=True, reduce_len=True).tokenize(text)]
	return stemmed_text

'''
Return true if the feature match one of the item in the list 'to_find'
'''
def bl_rt(word, to_find, stemmer = SnowballStemmer('english')):
    for ft in to_find:
        stemmed = stemmer.stem(ft)
        if (word in stemmed or stemmed in word):
            return True
    return False

def contain_a_word(tweet_text, bursty_list, stemmer = SnowballStemmer('english')):
    for ft in bursty_list:
        st = stemmer.stem(ft)
        if (st in tweet_text or ft in tweet_text):
            return True
    return False

'''
These two function are used to print the distribution of the word ove 100 samples, used for the debug of the burst detection module
'''
def binom_help(n_samples, probability):
    x = []
    y = []
    for i in range(0, n_samples+1):
        x.append(i)
        y.append(binom.pmf(i, n_samples, probability))
    return (x,y)

def plot_graph(N, expected, fixed, prob_win):
    a,b = binom_help(N, expected)
    c = fixed
    d = binom.pmf(fixed, N, expected)
    e = fixed
    f = prob_win
    plt.plot(a,b, 'ro')
    plt.plot(c,d, 'b^')
    plt.plot(e,f, 'gs')

    plt.show()

'''
Function used to split the index file, into smaller chunks used to retrieve tweets from different days
'''
def Split_Index_Per_Day(index_name, folder_name):
    
    if not path.exists(folder_name):
        makedirs(folder_name)
    
    with open(index_name) as index:
        for line in index:
            date = parser.parse(line.split('\t')[1])
            f_name = folder_name + "/" +str(date).split(" ")[0] + ".txt"
            fd = open(f_name, mode = 'a')
            print(line, end = '', file = fd)
            fd.close()

'''
From a file of the dataset, creates a list of time windows
'''
def create_windows(file_name_ds, window_size_minutes = 1):
    ret = []
    
    time_window = timedelta(minutes = window_size_minutes)

    try:
        database_fd = open(file_name_ds, mode = 'r')
        database = json.load(database_fd)
        database_fd.close()
    except:
        return ret

    date_id_list = [(parser.parse(database[tweet]['created_at']),database[tweet]['id_str']) for tweet in database]
    date_id_list.sort()

    i = 0
    database_size = len(date_id_list)-1
    
    first_date = date_id_list[i][0]
    first_tweet = date_id_list[i][1]
    
    curr_date = first_date
    curr_tweet = first_tweet

    while (i < database_size):
        window_tweets = list()
        while (curr_date - first_date < time_window and i < database_size):
            window_tweets.append(database[curr_tweet])
            i += 1
            curr_date = date_id_list[i][0]
            curr_tweet = date_id_list[i][1]
        ret.append(window_tweets)

        first_date = date_id_list[i][0]
        first_tweet = date_id_list[i][1]
    return ret