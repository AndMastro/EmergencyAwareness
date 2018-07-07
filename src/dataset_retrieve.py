import json
import time
import tweepy
import random

from os.path import join
from utils import dump_json_to_file, get_files_in_dir, Split_Index_Per_Day

'''
Builds the files representing our dataset, we use the path of linux
'''
def build_dataset_file(input_file, log_dir, output_dir, twitter_api):
    
    #get the file name in it's directory
    file_name = input_file.split('/')[-1]
    
    #read dataset from output_directory //it should fit in main memory with no problem
    out_path = join(output_dir, file_name)
    
    try:
        dataset_fd = open(out_path, mode = 'r')
        dataset = json.load(dataset_fd)
        dataset_fd.close()
    except FileNotFoundError:
        dataset = dict()
        dump_json_to_file(out_path, dataset)
    
    #open log - to check what is the last downloaded tweet in the file
    log_path = join(log_dir, file_name)
    
    try:
        log_fd = open(log_path, mode = 'r')
        last_line_read = json.load(log_fd)
        log_fd.close()
    except FileNotFoundError:
        last_line_read = 0
        dump_json_to_file(log_path, last_line_read)
        
    current_row = 1
    #read file for each row from input file
    with open(input_file) as index_fd:
        for line in index_fd:
            
            #increase the row number to get in the last position in the file
            if (current_row < last_line_read):
                current_row += 1
                continue
            
            if (current_row % 300 == 0):
                print("Dumping File")
                last_line_read = current_row
                #dump dataset and log
                dump_json_to_file(out_path, dataset)
                dump_json_to_file(log_path, last_line_read)
            
            #Get the id of the tweet to retrieve
            tweet_id = line.split("\t")[0].split(":")[2]
              
            #retrieve the tweet
            try:
                retrieved_tweet = twitter_api.get_status(tweet_id, tweet_mode = "extended")
                dataset[tweet_id] = retrieved_tweet._json
            except tweepy.RateLimitError as rle:
                print(str(rle))
                print("Exiting, the dataset will be updated")
                break
            except tweepy.TweepError as e:
                code = e.api_code
                if not (code == 34 or code == 63 or code == 144 or code == 179):
                    print("Unexpected error, quitting...")
                    print(str(e))
                    break
                #print(str(e))
                #print("TweetLost:" + str(tweet_id))
            
            current_row += 1
            
        #file finished
        last_line_read = current_row
        dump_json_to_file(out_path, dataset)
        dump_json_to_file(log_path, last_line_read)
    print("Quit:" + file_name)

'''
print the count of tweets retrieved in dataset
'''
def count_tweets(dataset_path, log_path):
    files = get_files_in_dir(dataset_path)
    N_tweets = 0

    for file in files:
        file_ds = open(join(dataset_path, file), mode = 'r')
        ds = json.load(file_ds)
        N_tweets += len(ds)
        file_ds.close()
    print('Collected: ' + str(N_tweets))

    files = get_files_in_dir(log_path)
    N_tweets = 0

    for file in files:
        file_ds = open(join(log_path, file), mode = 'r')
        ds = json.load(file_ds)
        N_tweets += ds
        file_ds.close()
    print('Listed: ' + str(N_tweets))


'''
Retrieve the tweets for each file of the dataset.
We chose the file at random in order tweets from every day of the dataset
'''
def retrieve_tweets(dataset_path, log_path, index_path, apis, from_main = False, main_index = ""):

    if from_main:
        Split_Index_Per_Day(main_index, index_path)
    
    files = get_files_in_dir(index_path)
    api = apis[0]

    num = len(apis)

    n = len(files)

    for i in range(0, n):
        file = random.choice(files)
        files.remove(file)
        if (api == apis[-1]):
            print("Wait for a while")
            time.sleep(240)
            api = apis[i % num]
        else:
            api = apis[i % num]
        print("----------------\n",join(index_path,file))
        build_dataset_file(join(index_path,file),log_path,dataset_path,api)

#Loading twitter APIs

um_consumer_key = "SCGLGW0n7g6LbFqT4IZj3HX27"
um_consumer_secret = "WaiJpI4kQuFWRLCE4jb5FH2aTwNX9TSi7CrOfVNO97wlYN1whV"
um_access_token = "996714080588222469-mygaT8BVVY5jKlhRubJgPiM2s8xw9d8"
um_access_token_secret = "ZOtB0TsAftlVCnGJi8C0dYxQgTKvNSDZLRcgk6uIGiI1B"
um_auth = tweepy.OAuthHandler(um_consumer_key, um_consumer_secret)
um_auth.set_access_token(um_access_token, um_access_token_secret)

am_consumer_key = "4yQ5nqYAqX5UNouZj4wP4mbLT"
am_consumer_secret = "OyJGOIf8cQRT4fkHmMa5hmlmH6cymIDSJPDYHkxQTYzCc8NIiO"
am_access_token = "219654730-JEaFbbhaMC3KPx8WUYcsHEXAPSq0hF7IjmnzU8Fb"
am_access_token_secret = "QBLhN9ex5KvoqKZI6UPD6XWIlmm2T7QdLHo9DkESirjOQ"
am_auth = tweepy.OAuthHandler(am_consumer_key, am_consumer_secret)
am_auth.set_access_token(am_access_token, am_access_token_secret)

um_api = tweepy.API(um_auth)
am_api = tweepy.API(am_auth)


apis = [um_api, am_api]

dataset_path = 'datasets/dataset'
log_path = 'datasets/logs'
index_path = 'datasets/days'
main_index = 'datasets/release.txt'

#retrieve_tweets(dataset_path, log_path, index_path, apis)
#count_tweets(dataset_path, log_path)