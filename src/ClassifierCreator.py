

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.decomposition import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import *
from FeatureExtractor import *
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score



def pipelineClassifier(bestTfIdf = False, bestWordEmb = False):

    ## Vectorization object
    vectorizer = TfidfVectorizer(strip_accents= None,
                                preprocessor = None,
                                )

    
    svm = SVC()
    pca = PCA(n_components=150)
    kBest = SelectKBest(k= 'all')
    svd = TruncatedSVD(n_components = 500)
    pipeline = None
    featureSelection = SelectPercentile(f_classif)
    

    if (not bestTfIdf and not bestWordEmb):
        print("No feature selection selected")
        pipeline = Pipeline([
            # Extract the subject & body
            #('subjectbody', SubjectBodyExtractor()),

            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for pulling tfidf
                    ('tf-idf', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('tfidf', TfidfVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                        #('best_features', svd),
                    ])),

                    ('date', Pipeline([
                        ('selector', FeatureSelector(index=2)),
                        ('oneHotEncodingDate', OneHotDate(buckNum = 50)),
                        #('best_features', pca),
                    ])),

                    ('word_emb', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('wordEmbeddings', WordEmbeddings(avg = True)),
                        #('best_features', pca),
                    ])),

                    #('context', Pipeline([
                     #   ('get_context', GetContext()),
                     #   ('tfidf', CountVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                    #])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'tf-idf': 1,
                    'oneHotEncodingDate': 1,
                    'wordEmbeddings': 1,
                    #'context': 0.6,
                },
            )),

            # Use a SVC classifier on the combined features
            ('svc', SVC(kernel='linear', C=1)),
        ])
        
    if (bestTfIdf and not bestWordEmb):
        print("Feature selection for TfIdf selected")
        pipeline = Pipeline([
            

            # Use FeatureUnion to combine features
            ('union', FeatureUnion(
                transformer_list=[

                    
                    ('tf-idf', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('tfidf', TfidfVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                        ('best_features', svd),
                    ])),

                    ('date', Pipeline([
                        ('selector', FeatureSelector(index=2)),
                        ('oneHotEncodingDate', OneHotDate(buckNum = 50)),
                        #('best_features', pca),
                    ])),

                    ('word_emb', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('wordEmbeddings', WordEmbeddings(avg = True)),
                        #('best_features', pca),
                    ])),

                    #('context', Pipeline([
                     #   ('get_context', GetContext()),
                     #   ('tfidf', CountVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                    #])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'tf-idf': 1,
                    'oneHotEncodingDate': 1,
                    'wordEmbeddings': 1,
                    #'context': 0.6,
                },
            )),

            # Use a SVC classifier on the combined features
            ('svc', SVC(kernel='linear', C=1)),
        ])
        
    if (not bestTfIdf and bestWordEmb):
        print("Feature selection for Word Embeddings selected")
        pipeline = Pipeline([
            # Extract the subject & body
            #('subjectbody', SubjectBodyExtractor()),

            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for pulling tfidf
                    ('tf-idf', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('tfidf', TfidfVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                        #('best_features', svd),
                    ])),

                    ('date', Pipeline([
                        ('selector', FeatureSelector(index=2)),
                        ('oneHotEncodingDate', OneHotDate(buckNum = 50)),
                        #('best_features', pca),
                    ])),

                    ('word_emb', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('wordEmbeddings', WordEmbeddings(avg = True)),
                        ('best_features', pca),
                    ])),

                    #('context', Pipeline([
                     #   ('get_context', GetContext()),
                     #   ('tfidf', CountVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                    #])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'tf-idf': 1,
                    'oneHotEncodingDate': 1,
                    'wordEmbeddings': 1,
                    #'context': 0.6,
                },
            )),

            # Use a SVC classifier on the combined features
            ('svc', SVC(kernel='linear', C=1)),
        ])
        
    if (bestTfIdf and bestWordEmb):
        print("Feature selection for TfIdf and Word Embeddings selected")
        pipeline = Pipeline([
            # Extract the subject & body
            #('subjectbody', SubjectBodyExtractor()),

            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for pulling tfidf
                    ('tf-idf', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('tfidf', TfidfVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                        ('best_features', svd),
                    ])),

                    ('date', Pipeline([
                        ('selector', FeatureSelector(index=2)),
                        ('oneHotEncodingDate', OneHotDate(buckNum = 50)),
                        #('best_features', pca),
                    ])),

                    ('word_emb', Pipeline([
                        ('selector', FeatureSelector(index=1)),
                        ('wordEmbeddings', WordEmbeddings(avg = True)),
                        ('best_features', pca),
                    ])),

                    #('context', Pipeline([
                     #   ('get_context', GetContext()),
                     #   ('tfidf', CountVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                    #])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'tf-idf': 1,
                    'oneHotEncodingDate': 1,
                    'wordEmbeddings': 1,
                    #'context': 0.6,
                },
            )),

            # Use a SVC classifier on the combined features
            ('svc', SVC(kernel='linear', C=1)),
        ])

    
    return pipeline





def kFoldCrossVal(classifier, xTrain, yTrain, k):
    scorings = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy' ]
    scores = cross_validate(classifier, xTrain, yTrain, scoring=scorings, cv=k, return_train_score=False, n_jobs = -1)

    f1 = np.mean(scores['test_f1_macro'])
    precision = np.mean(scores['test_precision_macro'])
    recall = np.mean(scores['test_recall_macro'])
    accuracy = np.mean(scores['test_accuracy'])
    
    return [f1, precision, recall, accuracy]




def fitPredict(classifier, xTrain, xTest, yTrain, yTest, target_names):
    
    classifier.fit(xTrain, yTrain)

    yPredicted = classifier.predict(xTest)

    
    output_classification_report = metrics.classification_report(
                                        yTest,
                                        yPredicted,
                                        target_names=target_names)
    print("")
    print("----------------------------------------------------")
    print(output_classification_report)
    print("----------------------------------------------------")
    print("")

    print("Accuracy:")
    acc = metrics.accuracy_score(yTest, yPredicted, normalize=True)
    print(acc)

    # Compute the confusion matrix
    confusion_matrix = metrics.confusion_matrix(yTest, yPredicted)
    print("")
    print("Confusion Matrix: True-Classes X Predicted-Classes")
    print(confusion_matrix)
    print("")


def baselineClassifier():

    ## Vectorization object
    vectorizer = TfidfVectorizer(strip_accents= None,
                                preprocessor = None,
                                )

    
    svm = SVC()
    pca = PCA(n_components=150)
    kBest = SelectKBest(k= 'all')
    svd = TruncatedSVD(n_components = 500)
    pipeline = None
    featureSelection = SelectPercentile(f_classif)
    

    print("Baseline classifier, just tfidf")
    
    pipeline = Pipeline([
        
        ('union', FeatureUnion(
            transformer_list=[

                
                ('tf-idf', Pipeline([
                    ('selector', FeatureSelector(index=1)),
                    ('tfidf', TfidfVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                    #('best_features', svd),
                ])),


                #('context', Pipeline([
                 #   ('get_context', GetContext()),
                 #   ('tfidf', CountVectorizer(min_df=3, ngram_range = (1,1), tokenizer = stemming_tokenizer)),
                #])),

            ],

            # weight components in FeatureUnion
            transformer_weights={
                'tf-idf': 1,
                #'context': 0.6,
            },
        )),

        # Use a SVC classifier on the combined features
        ('svc', SVC(kernel='linear', C=1)),
    ])
        
    
    return pipeline

