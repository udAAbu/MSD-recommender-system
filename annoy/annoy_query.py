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

def querying(spark, model_path, tree_path, query_df, rank):
    start = time.time()
    query_df = spark.read.parquet(query_df)
    model = ALSModel.load(model_path)
    
    def get_nn(u):
        from annoy import AnnoyIndex
        tree = AnnoyIndex(int(rank), "dot")
        tree.load(tree_path)
        return tree.get_nns_by_vector(u, n = 500)

    user_factors = model.userFactors

    get_nn_udf = udf(get_nn, returnType=(ArrayType(IntegerType())))

    query_users = [row['user'] for row in query_df.select('user').distinct().collect()]
    user_factors = user_factors.filter(col("id").isin(query_users))
    
    print("Annoy query begins...")
    annoy_prediction = user_factors.withColumn("track", get_nn_udf(col("features")))
    predictions = annoy_prediction.select(col("id").alias("user"), "track").repartition("user")
   
    ground_truth = query_df.groupby("user").agg(collect_list('track').alias("ground_truth")).repartition("user")
    
    df_result = predictions.join(broadcast(ground_truth), on = "user", how = "inner")
    predictionAndLabels = df_result.rdd.map(lambda row: (row['track'], row['ground_truth']))
    
    metrics = RankingMetrics(predictionAndLabels)
    MAP = metrics.meanAveragePrecision
    print("MAP (annoy)", MAP)
    
    prec = metrics.precisionAt(500)
    print("Precision @ 500(annoy)", prec)
    
    end = time.time()
    print("Annoy query ends")
    print("Annoy query_time: ", end - start)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help = "path of ALS model")
    parser.add_argument("--tree_path", help = "path to save annoy trees")
    parser.add_argument("--rank", help = "rank")
    parser.add_argument("--query_df", help = "path to the query file")
    
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
    
    querying(spark, model_path, tree_path, query_df, rank)