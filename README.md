# Emergency Awareness On Social Media
## Web Information Retrieval Project

The aim of our project is the categorization
of disater-related tweets. Our work
is divided into 3 tasks. Firstly, we classify
disaster-related tweets into predefined
classes, based on the work by (Stowe et
al., 2016). Secondly, we perform bursts
detection (Fung et al., 2005) in order to
understand which tweets are more likely
to be related to hazard events and finally,
among those tweets, we perform an online
clustering in order to gather tweets into
groups, basing this part on the paper by
(Yin et al., 2012). The results of classification
and clustering can be used by emergency
managers to understand how people
are reacting during a disaster situation in
order take proper actions both during and
after the emergency.

### Classification

In order to perform classification of tweets among different classes run the module ClassificationMain.py.

By editing the parameters in the script, you have to specifiy which class to perform the classification on among "Relevant", for revelance calssification, and "Report", "Action", "Information", "Sentiment", "Movement", "Preparation" for finer-grained classification. Running the module as it is will by default perform classification on "Information".

### Clustering

In order to run the clustering of tweets run the module ClusteringMain.py

Editing the parameters in the script is possible to specify the dataset-file on which run the clustering (the two specified in the script are the labeled dataset, and the other is the one used to build the test set), the delta and the time interval (in seconds) to de-activate the cluster. It is also possible decide wheteher use tf-idf or Word Embeddings (the latter is the default), and whether filter the tweets using the burst detection module.
The script will generate a .csv with the clusters.
