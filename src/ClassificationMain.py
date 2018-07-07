
from ClassifierCreator import *
from DatasetGetter import *

#choose class between Relevant, Information, Action, Report, Sentiment, Movement, Preparation
[xTrain, xTest, yTrain, yTest] = datasetGetter("Information")
target_names = ["Non Information", "Information"]

#choose baseline classifier of all feature classifier w/ or w/out feature selection
cl = pipelineClassifier()
#cl = baselineClassifier()
scores = kFoldCrossVal(cl, xTrain, yTrain, 5)
print(scores)
fitPredict(cl, xTrain, xTest, yTrain, yTest, target_names)

