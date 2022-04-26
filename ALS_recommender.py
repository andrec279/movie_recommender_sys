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
from pyspark.sql.functions import row_number, lit, udf, col
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
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
    ratings_train = spark.read.csv(path_to_file + 'ratings_train.csv', header='true', schema=schema)
    ratings_val = spark.read.csv(path_to_file + 'ratings_val.csv', header='true', schema=schema)
    ratings_test = spark.read.csv(path_to_file + 'ratings_test.csv', header='true', schema=schema)
    
    #create ground truth rankings by user from validation set and test set
    windowval = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    
    # ratings_val = ratings_val.withColumn('rating_count', F.row_number().over(windowval))
    # ratings_val = ratings_val.filter(ratings_val.rating_count<=100)
    # ratings_val = ratings_val.groupby('userId').agg(collect_list('movieId'))
    
    ratings_test = ratings_test.withColumn('rating_count', F.row_number().over(windowval))
    ratings_test = ratings_test.filter(ratings_test.rating_count<=100)
    ratings_test = ratings_test.groupby('userId').agg(collect_list('movieId'))
    
    t_prep = time.time()
    print('Data preparation time (.csv): ', round(t_prep-t0, 3), 'seconds')
    
    # Fit ALS model
    regParams = [5e-3, 7e-3]
    ranks = [50, 100]
    maxIters = [5, 10]
    
    for i in range(len(regParams)):
        for j in range(len(ranks)):
    
            als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', 
                      maxIter=20, rank=ranks[j], regParam=regParams[i], coldStartStrategy='drop')
            als_fitted = als.fit(ratings_train)
            
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
            
            # Using best params, evaluate on validation set
            preds_val = CV_als_fitted.bestModel.transform(ratings_val)
            print('Predicted ratings on validation set ("predictions" column)')
            preds_val.show()
            
            # Get top 100 ordered predictions for each user
            window_preds_val = Window.partitionBy('userId').orderBy(F.col('prediction').desc())
            window_truth_val = Window.partitionBy('userId').orderBy(F.col('rating').desc())
            
            # Get the true rank-ordered list of movieIds for each user
            preds_val = preds_val.withColumn('rating_count', F.row_number().over(window_preds_val))
            preds_val = preds_val.filter(preds_val.rating_count<=100)
            preds_val = preds_val.groupby('userId').agg(collect_list('movieId')).alias('true_ranking')
            
            # Get the predicted rank-ordered list of movieIds for each user
            truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth_val))
            truth_val = truth_val.filter(truth_val.rating_count<=100)
            truth_val = truth_val.groupby('userId').agg(collect_list('movieId')).alias('pred_ranking')
            
            # Join truth and predictions and evaluate
            preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
                                  .select(col('true_ranking'), col('pred_ranking'))\
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
     
    if local_source == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 