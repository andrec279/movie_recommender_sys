#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:16:08 2022

@author: andrechen
"""

#Use getpass to obtain user netID
import getpass

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import collect_list
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, row_number, monotonically_increasing_id
from pyspark.sql.types import IntegerType, ArrayType

def load_and_prep_ratings(path_to_file, spark, netID=None):
    
    ratings_train = spark.read.parquet(path_to_file + f'ratings_train{size}.parquet', header='true')
    ratings_val = spark.read.parquet(path_to_file + f'ratings_val{size}.parquet', header='true')
    ratings_test = spark.read.parquet(path_to_file + f'ratings_test{size}.parquet', header='true')
    
    ratings_train = ratings_train.drop('timestamp')
    ratings_val = ratings_val.drop('timestamp')
    ratings_test = ratings_test.drop('timestamp')

    #cast movieId and userId to ints and rating to float
    ratings_train = ratings_train.withColumn('movieId', ratings_train['movieId'].cast('integer'))
    ratings_train = ratings_train.withColumn('userId', ratings_train['userId'].cast('integer'))
    ratings_train = ratings_train.withColumn('rating', ratings_train['rating'].cast('float'))

    ratings_val = ratings_val.withColumn('movieId', ratings_val['movieId'].cast('integer'))
    ratings_val = ratings_val.withColumn('userId', ratings_val['userId'].cast('integer'))    
    ratings_val = ratings_val.withColumn('rating', ratings_val['rating'].cast('float')) 

    ratings_test = ratings_test.withColumn('movieId', ratings_test['movieId'].cast('integer'))
    ratings_test = ratings_test.withColumn('userId', ratings_test['userId'].cast('integer'))
    ratings_test = ratings_test.withColumn('rating', ratings_test['rating'].cast('float'))
    
    return ratings_train, ratings_val, ratings_test

def eval_ALS(truth_val, preds_val):
    preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
                          .select(col('true_ranking'), col('recommendations'))\
                          .rdd
    
    eval_metrics = RankingMetrics(preds_truth)
    mean_avg_precision = eval_metrics.meanAveragePrecision
    
    return mean_avg_precision

def main(spark, netID=None):
    if local_source == False:
        path_to_file = f'hdfs:/user/{netID}/'
    else:
        path_to_file = ''
    
    ratings_train, ratings_val, ratings_test = load_and_prep_ratings(path_to_file, spark, netID)
    
    window_truth = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    
    truth_test = ratings_test.withColumn('rating_count', F.row_number().over(window_truth))
    truth_test = truth_test.filter(truth_test.rating_count<=100)
    truth_test = truth_test.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    ALS_coldStart_recs = spark.read.parquet(path_to_file + 'ALS_coldStart_user_recs.parquet', header='true')
    ALS_coldStart_recs = ALS_coldStart_recs.withColumnRenamed('recs', 'recommendations')
    
    test_map = eval_ALS(truth_test, ALS_coldStart_recs)
    
    print('ALS with Cold Start MAP Score (test set):', test_map)

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    local_source = False # For local testing
    full_data = True
    size = '-small' if full_data == False else ''
     
    if local_source == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 