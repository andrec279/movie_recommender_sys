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
import numpy as np
import time

def main(spark, netID=None):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    
    if local_source == False:
        path_to_file = f'hdfs:/user/{netID}/'
    else:
        path_to_file = ''
        
    t0 = time.time()
    #load train, val, test data into DataFrames
    schema = 'index INT, userId INT, movieId INT, rating FLOAT, timestamp INT'
    ratings_train = spark.read.csv(path_to_file + f'ratings_train{size}.csv', header='true', schema=schema)
    ratings_val = spark.read.csv(path_to_file + f'ratings_val{size}.csv', header='true', schema=schema)
    ratings_test = spark.read.csv(path_to_file + f'ratings_test{size}.csv', header='true', schema=schema)
    
    # Get the predicted rank-ordered list of movieIds for each user
    window_truth_val = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth_val))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    t_prep = time.time()
    print('Data preparation time (.csv): ', round(t_prep-t0, 3), 'seconds')
    
    # Fit ALS model
    regParams = [5e-3, 7e-3]
    ranks = [50, 100]
    maxIters = [5, 10]
    
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')
    
    param_grid = ParamGridBuilder()\
                    .addGrid(als.maxIter, maxIters)\
                    .addGrid(als.rank, ranks)\
                    .addGrid(als.regParam, regParams)\
                    .build()
    
    # Search parameter space for optimal maxIter, rank, and regularization parameter
    # Here we use RMSE of predicting rating to tune hyperparameters, even though
    # we evaluate the final predictions on the validation set using only movie rankings
    evaluatorRMSE = RegressionEvaluator(metricName='rmse', labelCol='rating')
    CV_als = CrossValidator(estimator=als, 
                            estimatorParamMaps=param_grid,
                            evaluator=evaluatorRMSE, 
                            numFolds=5)
    CV_als_fitted = CV_als.fit(ratings_train)
    
    # Using best params, get top 100 recs from movies in training set and evaluate on validation set
    preds_val = CV_als_fitted.bestModel.recommendForAllUsers(100)
    
    # each set of user recommendations is in format [[movieId1, rating_pred1], [movieId2, rating_pred2], ...]
    # we can drop ratings and restructure to [movieId1, movieId2, ...]
    take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
    
    # Join truth and predictions and evaluate
    preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
                          .select(col('true_ranking'), col('recommendations'))\
                          .rdd

    eval_metrics = RankingMetrics(preds_truth)
    val_map = eval_metrics.meanAveragePrecision
    print('Validation MAP after model tuning: ', val_map)
    
    t_complete = time.time()
    print('\nTraining time (.csv) for {} configurations: {} seconds'.format(len(ranks)*len(regParams), round(t_complete-t_prep,3)))
    print('\nTotal runtime: {} seconds'.format(round(t_complete-t0, 3)))
    
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    local_source = True # For local testing
    full_data = False
    size = '-small' if full_data == False else ''
     
    if local_source == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 