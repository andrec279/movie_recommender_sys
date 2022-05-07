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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from functools import reduce
import numpy as np
import time
import random
import matplotlib.pyplot as plt

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

def fit_eval_ALS(spark, ratings_train, ratings_val):
    
    # Get the predicted rank-ordered list of movieIds for each user
    window_truth_val = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth_val))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', 
              coldStartStrategy='drop', rank=200, maxIter=10, regParam=0.005)

    als_model = als.fit(ratings_train)
    
    preds_val = als_model.recommendForAllUsers(100)
    take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
    preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
                          .select(col('true_ranking'), col('recommendations'))\
                          .rdd
    
    eval_metrics = RankingMetrics(preds_truth)
    val_map = eval_metrics.meanAveragePrecision
    
    return als_model, val_map

def fit_content_regression(spark, user_factors, item_factors, item_features, alphas):
    item_factors_features = item_factors.join(item_features, item_factors.id==item_features.movieId)\
                                        .drop('id')\
                                        .withColumnRenamed('features', 'target')\
    
    splits = item_factors_features.randomSplit([0.7, 0.3])
    train_features_data = splits[0]
    val_features_data = splits[1]
    X_ind = [str(i) for i in range(1,1129)]
    
    y_train = np.array(train_features_data.select('target').rdd.map(lambda row: np.array(row['target'])).collect())
    X_train = np.array(train_features_data.select(X_ind).collect())
    
    y_val = np.array(val_features_data.select('target').rdd.map(lambda row: np.array(row['target'])).collect())
    X_val = np.array(val_features_data.select(X_ind).collect())
    
    val_rmses = np.empty(len(alphas))
    train_rmses = np.empty(len(alphas))
    regressors = []
    
    print('Tuning regressor...')
    for i in range(len(alphas)):
        multi_regressor = MultiOutputRegressor(Ridge(alpha=alphas[i]))
        multi_regressor.fit(X_train, y_train)
        val_rmses[i] = mean_squared_error(y_val, multi_regressor.predict(X_val))
        train_rmses[i] = mean_squared_error(y_train, multi_regressor.predict(X_train))
        regressors.append(multi_regressor)
    print('Tuning done')
    
    best_rmse = np.min(val_rmses)
    best_alpha = alphas[np.argmin(val_rmses)]
    best_regressor = regressors[np.argmin(val_rmses)]
    
    print('Best alpha: ', best_alpha)
    print('Training RMSE: ', mean_squared_error(y_train, best_regressor.predict(X_train)))
    print('Training R2: ', r2_score(y_train, best_regressor.predict(X_train)))
    print('\n')
    print('Validation RMSE: ', best_rmse)
    print('Validation R2: ', r2_score(y_val, best_regressor.predict(X_val)))
    
    return best_regressor
    
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
    
    
    als_model, full_als_map = fit_eval_ALS(spark, ratings_train, ratings_val)
    
    # Get user and item factors
    user_factors = als_model.userFactors
    item_factors = als_model.itemFactors
    tag_genome = spark.read.parquet(path_to_file + 'tag_genome_pivot.parquet', header='true')
    
    alphas = np.array([0.1, 1, 10, 25, 50, 75, 100])
    content_regressor = fit_content_regression(spark, user_factors, item_factors, tag_genome, alphas)
                                            
    
    'TO-DO: evaluation section of full content cold start model'
    
    
    
    # # each set of user recommendations is in format [[movieId1, rating_pred1], [movieId2, rating_pred2], ...]
    # # we can drop ratings and restructure to [movieId1, movieId2, ...]
    # take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    # preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
    
    # # Join truth and predictions and evaluate
    # preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
    #                       .select(col('true_ranking'), col('recommendations'))\
    #                       .rdd

    # eval_metrics = RankingMetrics(preds_truth)
    # val_map = eval_metrics.meanAveragePrecision
    # print('Validation MAP after model tuning: ', val_map)
    
    # t_complete = time.time()
    # print('\nTraining time (.csv) for {} configurations: {} seconds'.format(len(ranks)*len(regParams)*len(maxIters), round(t_complete-t_prep,3)))
    # print('\nTotal runtime: {} seconds'.format(round(t_complete-t0, 3)))

 
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
