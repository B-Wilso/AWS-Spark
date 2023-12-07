from pyspark.context import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.sql import SQLContext
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

sc = SparkContext(appName="MLP")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

#Read libSVM-formatted iris dataset into pyspark.sql.DataFrame object.
data = sqlContext.read.format("libsvm").load("iris/iris_svm.txt")

#60/40 train/test split of iris dataset.
splits = data.randomSplit([0.6,0.4])
train = splits[0]
test = splits[1]

#Define layers for MLP Model. 4-node input layer, 5-node hidden layer, 4-node hidden layer, 3-node output layer.
layers = [4, 5, 4, 3]

##train and run model 10 different times to get time trials
#for x in range(10):

#Create MLP Model. maxIter specifies number of iterations to train on, layers specifies layers of model, blockSize specifies size of blocks used for stacking input data in matrices.
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128)

#Train the model using train data split, and time how long it takes to train.
t1 = time.time()
model = trainer.fit(train)
t2 = time.time()
    
#Make predictions on test data.
result = model.transform(test)
predictionAndLabels = result.select("prediction","label")
    
#Evaluate MLP predictions using Precision, Recall, and F1-Score
eval_pre = MulticlassClassificationEvaluator(metricName="precision")
eval_recall = MulticlassClassificationEvaluator(metricName="recall")
eval_f1 = MulticlassClassificationEvaluator(metricName="f1")
    
#Print evaluation and timing results
print("Precision: " + str(eval_pre.evaluate(predictionAndLabels)))
print("Recall: " + str(eval_recall.evaluate(predictionAndLabels)))
print("F1 Score : " + str(eval_f1.evaluate(predictionAndLabels)))
print("Train Time : %f s " % (t2-t1))
