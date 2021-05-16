#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

import numpy
# And pyspark.sql to get the spark session
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, QuantileDiscretizer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import argparse

def main(spark):
    df_train = spark.read.parquet("hdfs:/user/zn2041/df_train_clean.parquet")
    
    # modelling ALS
    rank = [20, 30, 40, 50]
    regParam = [0.01, 0.1, 1, 10]
    alpha = [5, 10, 15]
    
    for r in rank:
        for reg in regParam:
            for a in alpha:
                als = ALS(rank = r, maxIter=20, regParam=reg, userCol="user", itemCol="track", ratingCol="count",\
                        nonnegative = True, implicitPrefs = True, coldStartStrategy="drop", alpha = a, seed=42)
            
                model = als.fit(df_train)
                print(f"finished training ALS model with rank{r} and reg{reg} and alpha {a}")
                model.write().overwrite().save(f"hdfs:/user/zn2041/ALS_model_rank{r}_reg{reg}_alpha{a}")

    spark.stop()
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    config = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), \
                                         ('spark.driver.memory', '8g'), \
                                         ('spark.blacklist.enabled', False), \
                                         ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'), \
                                         ('spark.sql.autoBroadcastJoinThreshold', 100 * 1024 * 1024)])
    
    spark = SparkSession.builder.appName('training').config(conf=config).getOrCreate()

    # Call our main routine
    main(spark)
