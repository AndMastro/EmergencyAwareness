
# coding: utf-8

# In[5]:


from ClassifierCreator import *
from DatasetGetter import *


# In[ ]:


#choose class between Relevant, Information, Action, Report, Sentiment, Movement, Preparation
[xTrain, xTest, yTrain, yTest] = datasetGetter("Information")
target_names = ["Non Information", "Information"]


# In[ ]:


cl = pipelineClassifier()
#cl = baselineClassifier()
scores = kFoldCrossVal(cl, xTrain, yTrain, 5)
print(scores)
fitPredict(cl, xTrain, xTest, yTrain, yTest, target_names)

