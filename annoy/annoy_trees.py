import sys, argparse
import random
import time

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel

from annoy import AnnoyIndex
from tqdm import tqdm

def build_trees(spark, model_path, rank, num_trees, tree_path):
    model = ALSModel.load(model_path)
    
    item_factors = model.itemFactors
    annoy_item_factors = item_factors.withColumnRenamed("id", "annoy_id")
    
    tree = AnnoyIndex(int(rank), 'dot')
    print("Creating index...")
    for item in tqdm(annoy_item_factors.collect()):
        tree.add_item(item['annoy_id'], item['features'])
    
    n_trees = int(num_trees)
    print("Creating trees...")
    tree.build(n_trees)
    tree.save(tree_path)
    print("Finished!")
    
    spark.stop()

if __name__ == "__main__":
    spark = SparkSession.builder.appName("annoy_trees").getOrCreate()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help = "path of ALS model")
    parser.add_argument("--rank", help = "rank")
    parser.add_argument("--tree_path", help = "path to save annoy trees")
    parser.add_argument("--num_trees", help = "number of trees")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    rank = args.rank
    tree_path = args.tree_path
    num_trees = args.num_trees

    build_trees(spark, model_path, rank, num_trees, tree_path)