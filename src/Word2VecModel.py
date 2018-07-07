import gensim
import json
from os import listdir
from tokenLib import *

fileList = listdir("datasets/bigDataset")

xTrain = []

for file in fileList:
    datasetRaw = open("datasets/bigDataset/" + file, mode = "r")
    jsonDataset = json.load(datasetRaw)
    datasetRaw.close()
    
    for tweet in jsonDataset:
        xTrain.append(tweet_tokenizer(jsonDataset[tweet]["full_text"]))

print("Dataset created")

model = gensim.models.Word2Vec(xTrain, size = 200, window = 5, workers = 4, negative = 5, iter = 5)
w2v = dict(zip(model.wv.index2word, model.wv.vectors))
print("Model created")

outputFile = open("models/sockModel.txt", "w+", encoding = "utf-8")
for word in w2v:
    if "." not in str(word):
        outputFile.write(str(word))
        for elem in w2v[word]:
            outputFile.write(" " + str(elem))
        outputFile.write("\n")
outputFile.close()

print("Model saved")

print("Let's try the model, it recognises even emoji that appear in tweets")
print("Similar to :)")
model.wv.most_similar(positive=[':)'])

