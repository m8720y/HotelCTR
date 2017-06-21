# -*- coding: UTF-8 -*-
import sys
import pandas as pd
import matplotlib.pyplot as plt
from time    import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.tuning     import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree       import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics, MulticlassMetrics
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
    

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="/home/m8720y/Documents/ML/HotelCTR/"
    else:   
        Path="s3n://mollybigdata2017/"

#如果您要在cluster模式執行(hadoop yarn 或Spark Stand alone)，請先上傳檔案至HDFS目錄

def extract_label(record):
    label = (record[-1])
    return float(label)

def convert_float(x):
    return (0 if x == "NULL" else float(x))

def extract_features(record, featureEnd):
    features = [convert_float(field)  for  field in record[0: featureEnd-2]]
    return features

def PrepareData(sc, fileName):
    #1.import CSV file
    print("Start import training data .csv ...")
    dataFile = sc.textFile(Path + fileName)
    print "dataFile"
    header   = dataFile.first()
    print "header"
    data     = dataFile.filter(lambda line: line != header)
    lines    = data.map(lambda line: line.split(","))
    
    #2.convert to labeled point format
    labelpointRDD = lines.map(lambda r: \
                    LabeledPoint(extract_label(r), \
                    extract_features(r,len(r) - 1)))


    print("Total: " + str(lines.count()) + " data")
    print labelpointRDD.first()
    return labelpointRDD

def CrossValidation(trainData, kfold, sc):
    print("\n========== Cross Validation ===========")

    datasets = trainData.randomSplit([1] * kfold, seed = 123)
    avgTP = 0
    avgTN = 0
    avgFP = 0
    avgFN = 0

    for i in range(0, kfold):
        validation = datasets[i]
        train = sc.emptyRDD()
        for j in range(0, kfold):
            if i != j: 
                train = train.union(datasets[j])
        model = TrainModel(train)
        (tp, tn, fp, fn) = EvaluateModel(model, validation, sc)
        avgTP += tp
        avgTN += tn
        avgFP += fp
        avgFN += fn

    #ShowConfusionMatrix(avgTP, avgTN, avgFP, avgFN)




def TrainModel(trainData):
    print("\n========== Train Model ==============")
    startTime = time()

    model = LogisticRegressionWithLBFGS.train( \
            trainData, iterations = 10, \
            numClasses = 2)

    duration = time() - startTime
    print ("Time: " + str(duration))
    model.setThreshold(0.5)
    
    print("Threshold = " + str(model.threshold))
    print("Coefficients = " + str(model.weights))

    return model

def EvaluateModel(model, validationData, sc):
    print("\n========== Evaluate Model ===========")
    # Compute raw scores on the test set
    predictionAndLabels = \
    validationData.map(lambda lp: \
    (float(model.predict(lp.features)), lp.label))

    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(predictionAndLabels)

    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)

    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)

    #sqlContext = SQLContext(sc)
    #df = sqlContext.createDataFrame(predictionAndLabels, ["prediction", "answer"])
    #tp = df[(df.answer == 1) & (df.prediction == 1)].count()
    #tn = df[(df.answer == 0) & (df.prediction == 0)].count()
    #fp = df[(df.answer == 0) & (df.prediction == 1)].count()
    #fn = df[(df.answer == 1) & (df.prediction == 0)].count()
    tp = 1
    tn = 1
    fp = 1
    fn = 1

    # Instantiate metrics object
   
    #ShowConfusionMatrix(tp, tn, fp, fn)
    
    return (tp, tn, fp, fn)

    

def ShowConfusionMatrix(tp, tn, fp, fn):
    
    precision = float(tp)/float(tp+fp)
    recall    = float(tp)/float(tp+fn)
    TNR       = float(tn)/float(tn+fp)
    accuracy  = float(tp+tn)/float(tp+tn+fp+fn)

    print("+---+-----+-----+")
    print("|   |  y  |  n  | <- predicted")
    print("+---+-----+-----+")
    print("| Y |%5d+%5d|" % (tp, fn))
    print("+---+-----+-----+")
    print("| N |%5d+%5d|" % (fp, tn))
    print("+---+-----+-----+")

    print("       TP = %s" % tp)
    print("       TN = %s" % tn)
    print("       FP = %s" % fp)
    print("       FN = %s" % fn)

    print("      TNR = %s" % TNR)
    print("   Recall = %s" % recall)
    print("Precision = %s" % precision)
    print(" Accuracy = %s" % accuracy)

def CreateSparkContext():
    sparkConf = SparkConf()                                            \
                         .setAppName("RunDecisionTreeRegression")           \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)


if __name__ == "__main__":
    print("Logistic Regression")
    sc = CreateSparkContext()
    
    print("\n========== Prepare Data =============")
    trainData = PrepareData(sc, "trainP2000000.csv")
    testData  = PrepareData(sc, "test2000000.csv")
    trainData.persist(); testData.persist();

    print("\n========== Prepare Data =============")
    CrossValidation(trainData, 5, sc)

    print("\n========== Predict Model =============")
    model = TrainModel(trainData)
    EvaluateModel(model, testData, sc)
    
   
