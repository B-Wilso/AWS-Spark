from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SQLContext
from pyspark import SparkContext
import time

sc = SparkContext(appName="RandomForest")

sqlContext = SQLContext(sc)

data = sqlContext.read.format("libsvm").load("iris/iris_svm.txt")

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

(trainingData, testData) = data.randomSplit([0.6, 0.4])



rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")


# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
t1 = time.time()
model = pipeline.fit(trainingData)
t2 = time.time()

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show()

# Select (prediction, true label) and compute test error
f1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
r = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="recall")
p = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="precision")

print("Precision: %f" % p.evaluate(predictions) + "\nRecall: %f" % r.evaluate(predictions) + "\nF1: %f" % f1.evaluate(predictions))
print("Time Taken: " + str(t2-t1))
rfModel = model.stages[2]
print(rfModel)  # summary only
