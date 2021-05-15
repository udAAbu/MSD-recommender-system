# MSD-recommender-system
Recommendation System For Million Song Dataset

# Code documentation

## Baseline
The baseline folder contains 3 python files:
1. [Preprocessing.py](https://github.com/udAAbu/MSD-recommender-system/blob/main/baseline/preprocessing.py). This is the script we used for cleaning up the datasets so that they can be used to fit into pyspark ALS model for training and evaluation purpose. It will produce three parquet files, which are *df_train_clean.parquet, df_val_clean.parquet, df_test_clean.parquet*, and they will be saved to HDFS.
2. [train_model.py](https://github.com/udAAbu/MSD-recommender-system/blob/main/baseline/train_model.py). This is the training script. The ALS models will be trained based on the *df_train_clean.parquet*, and the model will also be saved to HDFS.
3. [eval_model.py](https://github.com/udAAbu/MSD-recommender-system/blob/main/baseline/eval_model.py). This is the evaluation script. You can decide to evaluate the model on the validation set or test set by specifying the parquet file path when you submit the job. 

  - Here is a simple example showing the precedures to run the codes (evaluate on validation set):
    ```
    spark-submit preprocessing.py
    spark-submit train_model.py
    spark-submit eval_model.py hdfs:/user/{netID}/df_val_clean.parquet
    ```
4. [tuning_results.xlsx](https://github.com/udAAbu/MSD-recommender-system/blob/main/baseline/tuning_results.xlsx). This is a spreadsheet showing the results for all the hyperparameters we tried. 

## Annoy
To successfully run the codes in this part, you need to export the model parameters from HDFS to your local machine, as well as the *df_val_clean.parquet* and *df_test_clean.parquet* file depending on how you filter the user factors. This annoy folder contains 2 files:
1. [annoy_trees.py](https://github.com/udAAbu/MSD-recommender-system/blob/main/annoy/annoy_trees.py). This is the script that creates the index and builds the trees. You need to specify the model's path in your local machine, the number of rank you used, the number of trees you want to build, and where to save the trees. 
2. [annoy_query.py](https://github.com/udAAbu/MSD-recommender-system/blob/main/annoy/annoy_query.py). This is the script that does the query and evaluate the performance. Just as before, you need to specify the model's path in your local machine, the number of rank you used, the path of the built trees, and which file you want to query on. 
    
  - Here is a simple example showing the procedures to run the codes (query on validation set):
    ```
    python annoy_trees.py --model_path ..\models\ALS_model_rank50_reg1_alpha15\ --rank 50 --tree_path .\trees\annoy_tree_25.ann --num_trees 25
    python annoy_query.py --model_path ..\models\ALS_model_rank50_reg1_alpha15\ --rank 50 --tree_path .\trees\annoy_tree_25.ann --query_df ..\data\df_val_clean.parquet\ 
    ```
3. 
