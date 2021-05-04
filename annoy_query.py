import sys, argparse
import random
import time

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics

from annoy import AnnoyIndex
from tqdm import tqdm
from pyspark.sql.types import *

def querying(spark, model_path, tree_path, query_df, rank, annoy = None):
    query_df = spark.read.parquet(query_df)
    '''
    df_train = spark.read.parquet("df_train_clean.parquet")
    df_val = spark.read.parquet("df_val_clean.parquet")
    df_test = spark.read.parquet("df_test_clean.parquet")
    '''
    print(query_df.printSchema())
    model = ALSModel.load(model_path)
    
    if annoy == 'True':
        def find_candidates(u):
            from annoy import AnnoyIndex
            tree = AnnoyIndex(int(rank), "dot")
            tree.load(tree_path)
            return tree.get_nns_by_vector(u, n = 500)
        
        user_factors = model.userFactors

        find_candidates_udf = udf(find_candidates, returnType=(ArrayType(IntegerType())))

        #start querying
        print("Annoy query begins")
        start = time.time()

        annoy_candidates = user_factors.withColumn("predictions", find_candidates_udf(col("features")))
        prediction = annoy_candidates.select(col("id").alias("user"), "predictions")
        prediction_list =  [{'user': row['user'], 'predictions': row['predictions']} for row in prediction.collect()]
        
        end = time.time()
        print("Annoy query ends")
        print("Annoy query_time: ", end - start)
        
        prediction = annoy_candidates.select(col("id").alias("user"), "predictions")
        ground_truth = query_df.groupby("user").agg(collect_list('item').alias("ground_truth"))

        df_result = prediction.join(ground_truth, on = "user", how = "inner")
        predictionAndLabels = df_result.rdd.map(lambda row: (row['predictions'], row['ground_truth']))
        
        #compute metrics
        metrics = RankingMetrics(predictionAndLabels)

        MAP = metrics.meanAveragePrecision
        print("MAP(annoy): ", MAP)

        prec = metrics.precisionAt(500)
        print("Precision @ 500(annoy)", prec)
    else:
        print("Brute-force query begins")
        start = time.time()
        
        query_users = query_df.select("user").distinct()
        predictions = model.recommendForUserSubset(query_users, 500).select('user', 'recommendations.item')
        predictions_list =  [{'user': row['user'], 'predictions': row['item']} for row in predictions.collect()]

        end = time.time()
        print("Brute-force query ends")
        print("Brute-force query_time: ", end - start)
        
        ground_truth = query_df.groupby("user").agg(collect_list('item').alias("ground_truth"))
        df_result = predictions.join(ground_truth, on = 'user', how = 'inner')

        predictionAndLabels = df_result.rdd.map(lambda row: (row['item'], row['ground_truth']))

        metrics = RankingMetrics(predictionAndLabels)

        MAP = metrics.meanAveragePrecision
        print("MAP(brute-force): ", MAP)

        prec = metrics.precisionAt(500)
        print("Precision @ 500(brute-force)", prec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help = "path of ALS model")
    parser.add_argument("--tree_path", help = "path to save annoy trees")
    parser.add_argument("--rank", help = "rank")
    parser.add_argument("--query_df", help = "path to the query file")
    parser.add_argument("--annoy", help = "use annoy or not", choices = ['True', 'False'])
    
    spark = SparkSession.builder.appName("query")\
    .config("spark.executor.memory", "16g")\
    .config("spark.driver.memory", "16g")\
    .config("spark.sql.shuffle.partitions", "50")\
    .getOrCreate()
    
    args = parser.parse_args()
    
    model_path = args.model_path
    rank = args.rank
    tree_path = args.tree_path
    query_df = args.query_df
    annoy = args.annoy
    
    querying(spark, model_path, tree_path, query_df, rank, annoy)