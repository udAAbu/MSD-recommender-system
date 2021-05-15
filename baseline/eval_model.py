#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
import numpy
import sys
# And pyspark.sql to get the spark session
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as fn 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, QuantileDiscretizer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import time

def main(spark, eval_df):
    start = time.time()

    df_val = spark.read.parquet(eval_df)

    # modelling ALS
    rank = [20, 30, 40, 50]
    regParam = [0.01, 0.1, 1, 10]
    alpha = [5, 10, 15]
    
    for r in rank:
        for reg in regParam:
            for a in alpha:
                model = ALSModel.load(f"hdfs:/user/zn2041/ALS_model_rank{r}_reg{reg}_alpha{a}")
                print(f"finished loading ALS_model_rank{r}_reg{reg}_alpha{a}")

                query_users = df_val.select("user").distinct()

                predictions = model.recommendForUserSubset(query_users, 500).select('user', 'recommendations.track').repartition("user")

                ground_truth = df_val.groupBy("user").agg(collect_list('track').alias("ground_truth")).repartition("user")

                df_result = predictions.join(broadcast(ground_truth), on = 'user', how = 'inner')

                predictionAndLabels = df_result.rdd.map(lambda row: (row['track'], row['ground_truth']))

                metrics = RankingMetrics(predictionAndLabels)

                MAP = metrics.meanAveragePrecision
                print("MAP(brute-force): ", MAP)
                prec = metrics.precisionAt(500)
                print("Precision @ 500(brute-force)", prec)
                end = time.time()
                print("total validation time: ", end - start)
    spark.stop()


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    config = pyspark.SparkConf().setAll([\
					('spark.executor.memory', '8g'),\
					('spark.driver.memory', '8g'),\
					('spark.blacklist.enabled', False),\
					('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'),\
					('spark.sql.autoBroadcastJoinThreshold', 100*1024*1024),\
					('spark.sql.broadcastTimeout', 300)])

    spark = SparkSession.builder.appName('ds_1004_project').config(conf=config).getOrCreate()
    eval_df = sys.argv[1]
    # Call our main routine
    main(spark, eval_df)

