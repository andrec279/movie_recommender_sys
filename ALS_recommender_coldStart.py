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
from pyspark.sql.functions import col, udf, row_number, monotonically_increasing_id
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

import dask
import dask.array as d

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

def fit_eval_ALS(spark, ratings_train, truth_val, truth_test):
    
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', 
              coldStartStrategy='drop', rank=150, maxIter=10, regParam=0.005)

    als_model = als.fit(ratings_train)
    
    preds_val = als_model.recommendForAllUsers(100)
    take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
    
    val_map = eval_ALS(truth_val, preds_val)
    test_map = eval_ALS(truth_test, preds_val)
    
    return als_model, val_map, test_map

def train_content_regressor(spark, item_factors_features, alphas, path_to_file):
    
    splits = item_factors_features.randomSplit([0.7, 0.3])
    train_features_data = splits[0].persist()
    val_features_data = splits[1].persist()
    X_ind = [str(i) for i in range(1,1129)]
    
    y_train = np.array(train_features_data.select('target').rdd.map(lambda row: np.array(row['target'])).collect())
    X_train = np.array(train_features_data.select(X_ind).collect())
    movieIds_train = np.array(train_features_data.select('movieId').collect())
    
    y_val = np.array(val_features_data.select('target').rdd.map(lambda row: np.array(row['target'])).collect())
    X_val = np.array(val_features_data.select(X_ind).collect())
    movieIds_val = np.array(val_features_data.select('movieId').collect())
    
    val_rmses = np.empty(len(alphas))
    train_rmses = np.empty(len(alphas))
    
    val_r2 = np.empty(len(alphas))
    train_r2 = np.empty(len(alphas))
    
    regressors = []
    
    print('Tuning regressor...')
    for i in range(len(alphas)):
        multi_regressor = MultiOutputRegressor(Ridge(alpha=alphas[i]))
        multi_regressor.fit(X_train, y_train)
        val_rmses[i] = mean_squared_error(y_val, multi_regressor.predict(X_val))
        train_rmses[i] = mean_squared_error(y_train, multi_regressor.predict(X_train))
        
        val_r2[i] = r2_score(y_val, multi_regressor.predict(X_val))
        train_r2[i] = r2_score(y_train, multi_regressor.predict(X_train))
        
        regressors.append(multi_regressor)
    print('Tuning done\n\n')
    
    best_rmse = np.min(val_rmses)
    best_alpha = alphas[np.argmin(val_rmses)]
    best_regressor = regressors[np.argmin(val_rmses)]
    
    print('Best alpha: ', best_alpha)
    print('Training RMSE: ', mean_squared_error(y_train, best_regressor.predict(X_train)))
    print('Training R2: ', r2_score(y_train, best_regressor.predict(X_train)))
    print('\n')
    print('Validation RMSE: ', best_rmse)
    print('Validation R2: ', r2_score(y_val, best_regressor.predict(X_val)))
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    
    axs[0].plot(alphas, val_rmses, label='rmse')
    axs[0].plot(alphas, val_r2, label='r2')
    axs[0].set_xlabel('regularization constant')
    axs[0].set_title('Validation Metrics')
    axs[0].legend()
    
    axs[1].plot(alphas, train_rmses, label='rmse')
    axs[1].plot(alphas, train_r2, label='r2')
    axs[1].set_xlabel('regularization constant')
    axs[1].set_title('Training Metrics')
    axs[1].legend()
    
    plt.savefig(path_to_file + 'content_model_metrics.png')
    
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
    
    # Get the predicted rank-ordered list of movieIds for each user
    window_truth = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    truth_test = ratings_test.withColumn('rating_count', F.row_number().over(window_truth))
    truth_test = truth_test.filter(truth_test.rating_count<=100)
    truth_test = truth_test.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    als_model, full_als_map = fit_eval_ALS(spark, ratings_train, truth_val, truth_test)
    
    # Get holdout set of movieIds, remove from training set, and train new ALS model (simulate cold start)
    item_factors = als_model.itemFactors.persist()
    movieIds_held_out_df = item_factors.sample(fraction=0.05, seed=1)
    movieIds_held_out = np.array(movieIds_held_out_df.select('id').collect()).flatten()
    
    ratings_train_cold = ratings_train.filter(~col('movieId').isin(movieIds_held_out.tolist()))
    cold_ALS_model, cold_map, cold_test_map = fit_eval_ALS(spark, ratings_train_cold, truth_val)
    
    print('Getting user / item factors from cold_ALS_model..')
    # Get new model's user / item factors
    user_factors_cold = cold_ALS_model.userFactors
    item_factors_cold = cold_ALS_model.itemFactors
    print('Got user and item factors df')
    
    tag_genome = spark.read.parquet('tag_genome_pivot.parquet', header='true')
    item_factors_train_genome = item_factors_cold.join(tag_genome, item_factors_cold.id==tag_genome.movieId)\
                                        .drop('id')\
                                        .withColumnRenamed('features', 'target')
                                        
    item_factors_test_genome = movieIds_held_out_df.join(tag_genome, movieIds_held_out_df.id==tag_genome.movieId)\
                                            .drop('id')\
                                            .drop('features')
    
    user_factors_cold.write.mode('overwrite').option('header', True).parquet(path_to_file + 'user_factors_cold.parquet')
    movieIds_held_out_df.write.mode('overwrite').option('header', True).parquet(path_to_file + 'movieIds_held_out.parquet')
    item_factors_train_genome.write.mode('overwrite').option('header', True).parquet(path_to_file + 'item_factors_train_genome.parquet')
    item_factors_test_genome.write.mode('overwrite').option('header', True).parquet(path_to_file + 'item_factors_test_genome.parquet')
    
    print('Done, wrote user_factors_cold, movieIds_held_out_df, item_factors_train_genome, item_factors_test_genome to parquet')
    print('Full ALS Test set MAP:', )
    print('Completed in {} seconds'.format(time.time() - t0))
 
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
