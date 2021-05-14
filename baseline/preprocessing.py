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
    train_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    val_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    test_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'

    df_train = spark.read.parquet(train_set_file_dir).repartition(100)
    df_val = spark.read.parquet(val_set_file_dir)
    df_test = spark.read.parquet(test_set_file_dir)
    
    df_train = df_train.drop("__index_level_0__")
    df_val = df_val.drop("__index_level_0__")
    df_test = df_test.drop("__index_level_0__")

    # convert user_id and track_id into numeric form
    user_indexer = StringIndexer(inputCol='user_id', outputCol='user', handleInvalid = 'skip')
    track_indexer = StringIndexer(inputCol='track_id', outputCol='track', handleInvalid = 'skip')
    
    ppl = Pipeline(stages=[user_indexer, track_indexer])
    ppl_transformer = ppl.fit(df_train)
    
    df_train = ppl_transformer.transform(df_train)
    df_val = ppl_transformer.transform(df_val)
    df_test = ppl_transformer.transform(df_test)
    
    df_train = df_train.withColumn('user', df_train.user.cast('integer')).withColumn('track', df_train.track.cast('integer'))
    df_val = df_val.withColumn('user', df_val.user.cast('integer')).withColumn('track', df_val.track.cast('integer'))
    df_test = df_test.withColumn('user', df_test.user.cast('integer')).withColumn('track', df_test.track.cast('integer'))

    df_train.select("track", "track_id").dropDuplicates().write.mode("overwrite").parquet("track_indexer")    

    df_train = df_train.drop("user_id").drop("track_id")
    df_val = df_val.drop("user_id").drop("track_id")
    df_test = df_test.drop("user_id").drop("track_id")    
    
    df_train.write.mode("overwrite").parquet("hdfs:/user/zn2041/df_train_clean.parquet")
    df_val.write.mode("overwrite").parquet("hdfs:/user/zn2041/df_val_clean.parquet")
    df_test.write.mode("overwrite").parquet("hdfs:/user/zn2041/df_test_clean.parquet")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    config = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), \
                                         ('spark.driver.memory', '8g'), \
                                         ('spark.blacklist.enabled', False), \
                                         ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'), \
                                         ('spark.sql.autoBroadcastJoinThreshold', 100 * 1024 * 1024)])
    
    spark = SparkSession.builder.appName('preprocessing').config(conf=config).getOrCreate()

    # Call our main routine
    main(spark)
