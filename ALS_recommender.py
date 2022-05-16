#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:05:05 2022

@author: andrechen
"""
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import collect_list
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from functools import reduce
import numpy as np
import time
import sys


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

def fit_eval_ALS(spark, ratings_train, ratings_val, params):
    
    # Get the predicted rank-ordered list of movieIds for each user
    window_truth = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    regParams = params['regParams']
    ranks = params['ranks']
    maxIters = params['maxIters']
    
    param_scores = {}
    models = []
    best_val_map = 0
    best_regParam = 0
    best_rank = 0
    best_maxIter = 0
    best_model_index = 0
    
    model_index=0
    for regParam in regParams:
        for rank in ranks:
            for maxIter in maxIters:
                als = ALS(userCol='userId', itemCol='movieId', 
                          ratingCol='rating', regParam=regParam, 
                          rank=rank, maxIter=maxIter, 
                          coldStartStrategy='drop')
                model = als.fit(ratings_train)
                models.append(model)
                
                param_key = f'regParam: {regParam}, rank: {rank}, maxIter: {maxIter}'
                
                '---{START} Validation portion of tuning---'
                preds_val = model.recommendForAllUsers(100)
                take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
                preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
                preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
                                      .select(col('true_ranking'), col('recommendations'))\
                                      .rdd
                
                eval_metrics = RankingMetrics(preds_truth)
                val_map = eval_metrics.meanAveragePrecision
                '---{END} Validation portion of tuning---'
                
                param_scores[param_key] = val_map
                
                if val_map > best_val_map:
                    best_val_map = val_map
                    best_regParam = regParam
                    best_rank = rank
                    best_maxIter = maxIter
                    best_model_index = model_index
                    print('best_model_index', best_model_index)
                
                model_index+=1
                
    best_model = models[best_model_index]
    print('Tuning results:')
    print('Param grid and MAP values: ')
    for key in param_scores:
        print((key, param_scores[key]))
    print('Best regParam: ', best_regParam)
    print('Best rank: ', best_rank)
    print('Best maxIter: ', best_maxIter)
    print('Best MAP on validation set: ', best_val_map)
    
    print('\n\n')
    
    print('Final MAP on test set: ')
    
    return best_model

def main(spark, netID=None):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    
    t0 = time.time()
    
    if local_source == False:
        path_to_file = f'hdfs:/user/{netID}/'
    else:
        path_to_file = ''    
     
    ratings_train, ratings_val, ratings_test = load_and_prep_ratings(path_to_file, spark, netID)
    
    t_prep = time.time()
    print('Data preparation time (.csv): ', round(t_prep-t0, 3), 'seconds')

    # Fit ALS model
    regParams = np.array([1e-2])
    ranks = np.array([150, 200, 500, 1000])
    maxIters = np.array([5, 10])
    
    params = {'regParams': regParams, 'ranks': ranks, 'maxIters': maxIters}
    
    als_model = fit_eval_ALS(spark, ratings_train, ratings_val, ratings_test, params)
    
    # Evaluate model on test set and print results
    window_truth = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_test = ratings_test.withColumn('rating_count', F.row_number().over(window_truth))
    truth_test = truth_test.filter(truth_test.rating_count<=100)
    truth_test = truth_test.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    preds_test = als_model.recommendForAllUsers(100)
    take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    preds_test = preds_test.withColumn('recommendations', take_movieId('recommendations'))
    
    test_map = eval_ALS(truth_test, preds_test)
    
    t_complete = time.time()
    print('\nTraining time (.csv) for {} configurations: {} seconds'.format(len(ranks)*len(regParams)*len(maxIters), round(t_complete-t_prep,3)))
    print('Final Test MAP: {}'.format(test_map))
    print('\nTotal runtime: {} seconds'.format(round(t_complete-t0, 3)))
    
    
 
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    local_source = False # For local testing
    full_data = False
    size = '-small' if full_data == False else ''
     
    if local_source == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 
