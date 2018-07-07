import json
from sklearn.model_selection import train_test_split


def datasetGetter(targetClass):
    #load dataset for specified class

    datasetRaw = open("datasets/dataset_merged.json", mode = "r")
    jsonDataset = json.load(datasetRaw)
    datasetRaw.close()

    xData = []
    yData = []
    
    lb = targetClass
    
    monthArray = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dic"]

    if lb != "Relevant":
        for tweet in jsonDataset:
            classes = jsonDataset[tweet]["annotations"]
            found = False
            date = jsonDataset[tweet]["created_at"].split(" ")
            formatDate = ""
            if len(date) == 2:
                formatDate = date[0] + " " + date[1]
            else:
                formatDate = str(date[5]) + "-" + str(monthArray.index(date[1])+1) + "-" +  str(date[2]) + " " + str(date[3])
            for cl in classes:
                if cl.find(lb) != -1:
                    found = True

                    xData.append([jsonDataset[tweet]["id"], jsonDataset[tweet]["text"], formatDate, jsonDataset[tweet]["previous"]])
                    yData.append(1)
                    break
            if not found and ['None'] != classes:
                xData.append([jsonDataset[tweet]["id"], jsonDataset[tweet]["text"], formatDate, jsonDataset[tweet]["previous"]])
                yData.append(0)

    else:
        for tweet in jsonDataset:
            classes = jsonDataset[tweet]["annotations"]
            date = jsonDataset[tweet]["created_at"].split(" ")

            formatDate = ""
            if len(date) == 2:
                formatDate = date[0] + " " + date[1]
            else:
                formatDate = str(date[5]) + "-" + str(monthArray.index(date[1])+1) + "-" +  str(date[2]) + " " + str(date[3])

            if classes == ['None']:
                xData.append([jsonDataset[tweet]["id"], jsonDataset[tweet]["text"], formatDate, jsonDataset[tweet]["previous"]])
                yData.append(0)

            else:
                xData.append([jsonDataset[tweet]["id"], jsonDataset[tweet]["text"], formatDate, jsonDataset[tweet]["previous"]])
                yData.append(1)

    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.25)

    numClass = 0
    numNotClass = 0
    for c in yTrain:
        if c == 1:
             numClass += 1
        else:
            numNotClass += 1

    target_names = ["Non"+ lb, lb]
    print("Number of samples in training set")
    print(target_names[1] + ": " + str(numClass))
    print(target_names[0] + ": " + str(numNotClass))

    print("")
    print("----------------------")
    print("Creating Training Set and Test Set")
    print("")
    print("Training Set Size")
    print(len(yTrain))
    print("")
    print("Test Set Size")
    print(len(yTest))
    print("")
    print("Classes:")
    print(target_names)
    print("----------------------")
    
    return xTrain, xTest, yTrain, yTest





