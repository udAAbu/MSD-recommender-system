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
import pyspark.sql.functions as fn
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, QuantileDiscretizer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

def create_train_subset():
    train_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_train_new.parquet'

    df_train = spark.read.parquet(train_set_file_dir)
    df_train = df_train.drop("__index_level_0__")
    df_train.createOrReplaceTempView('df_train')

    df_train_subset = spark.sql('SELECT * FROM df_train WHERE user_id IN (SELECT DISTINCT user_id FROM df_train LIMIT 500000)')
    df_train_subset.write.parquet('hdfs:/user/lc3424/train_subset.parquet')
    df_train_subset.createOrReplaceTempView('df_train_subset')

def main(spark):
    train_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    train_set_subset_file_dir = 'hdfs:/user/lc3424/train_subset.parquet'
    val_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    test_set_file_dir = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'

    # create_train_subset()

    df_train = spark.read.parquet(train_set_file_dir)
    df_train.createOrReplaceTempView('df_train')

    # convert user_id and track_id into numeric form
    user_indexer = StringIndexer(inputCol='user_id', outputCol='user')
    track_indexer = StringIndexer(inputCol='track_id', outputCol='track')
    ppl = Pipeline(stages=[user_indexer, track_indexer])
    ppl_transformer = ppl.fit(df_train)
    df_train = ppl_transformer.transform(df_train)
    df_train = df_train.drop("user_id").drop("track_id")

    # modelling ALS
    als = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="track", ratingCol="count", coldStartStrategy="drop", seed=42)
    model = als.fit(df_train)

    spark.stop()




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    config = pyspark.SparkConf().setAll([('spark.executor.memory', '4g'), ('spark.blacklist.enabled', False)])
    spark = SparkSession.builder.appName('ds_1004_project').config(conf=config).getOrCreate()

    # Call our main routine
    main(spark)
