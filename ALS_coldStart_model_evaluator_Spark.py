#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:36:54 2022

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
from sklearn.neighbors import NearestNeighbors

from functools import reduce
import numpy as np
import time
import random
import matplotlib.pyplot as plt

import dask
import dask.array as da
import dask.dataframe as dd

def eval_ALS(truth_val, preds_val):
    preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
                          .select(col('true_ranking'), col('recommendations'))\
                          .rdd
    
    eval_metrics = RankingMetrics(preds_truth)
    mean_avg_precision = eval_metrics.meanAveragePrecision
    
    return mean_avg_precision

def train_content_regressor(genome_cols, item_factors_features, alphas, path_to_file):
    print('Training content regressor...')
    splits = item_factors_features.randomSplit([0.7, 0.3])
    train_features_data = splits[0].persist()
    val_features_data = splits[1].persist()
    
    y_train = np.array(train_features_data.select('target').collect()).squeeze()
    X_train = np.array(train_features_data.select(genome_cols).collect()).squeeze()
    
    y_val = np.array(val_features_data.select('target').collect()).squeeze()
    X_val = np.array(val_features_data.select(genome_cols).collect()).squeeze()
    
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
    
    if local_source == False:
        path_to_file = f'hdfs:/user/{netID}/'
    else:
        path_to_file = ''
    
    # Read user factors, item factors, and held out movieIds as dask dataframes
    genome_cols = [str(i) for i in range(1,1129)]
    user_factors_cold = spark.read.parquet(path_to_file + 'user_factors_cold.parquet', header='true')
    item_factors_train_genome = spark.read.parquet(path_to_file + 'item_factors_train_genome.parquet', header='true')
    item_factors_test_genome = spark.read.parquet(path_to_file + 'item_factors_test_genome.parquet', header='true')
    
    movieIds_train = np.array(item_factors_train_genome.select('movieId').collect()).flatten()
    movieIds_test = np.array(item_factors_test_genome.select('movieId').collect()).flatten()
    
    print('Loading user / item factors into matrix..')
    user_factors_cold_matrix = np.array(user_factors_cold.select('features').collect()).squeeze()
    item_factors_cold_matrix = np.array(item_factors_train_genome.select('target').collect()).squeeze()
    print('Done')
    
    # Train / validate content model to use genome data (features) to predict item factors
    print('Training content model..')
    alphas = np.array([0.1, 1, 10, 25, 50, 75, 100])
    content_model = train_content_regressor(genome_cols, item_factors_train_genome, alphas, path_to_file)
    print('Done')
    
    # Join held out movies to their tag genome data, then predict their item factors using content model
    X_test = np.array(item_factors_test_genome.select(genome_cols).collect()).squeeze()
    held_out_factors_pred = content_model.predict(X_test)
    
    print('Creating combined items matrix..')
    # Combine item factors from ALS with item factors from content model
    item_factors_matrix_combined = np.vstack((item_factors_cold_matrix, held_out_factors_pred))
    # index of movie A in movieIds = row of movie A in item_factors_matrix_combined
    movieIds = np.concatenate((movieIds_train, movieIds_test))
    print('Done')
    
    # Evaluate cold start model
    print('Computing K most similar vectors')
    userIds = np.array(user_factors_cold.select('id').collect()).flatten()
    neigh = NearestNeighbors(n_neighbors=100, radius=1000, metric='cosine')
    neigh.fit(item_factors_matrix_combined)
    recs_indices = neigh.kneighbors(user_factors_cold_matrix, 100, return_distance=False)
    recs = np.array([movieIds[recs_indices[i]] for i in range(len(recs_indices))]).tolist()
    
    
    user_recs = spark.createDataFrame([(int(userIds[i]), recs[i]) for i in range(len(recs))], schema=['userId', 'recommendations'])
    print('Done\n')
    print('User Recs', user_recs.show(5))
    
    ratings_val = spark.read.parquet(path_to_file + f'ratings_val{size}.parquet', header='true')
    ratings_test = spark.read.parquet(path_to_file + f'ratings_test{size}.parquet', header='true')
    
    ratings_val = ratings_val.drop('timestamp')
    ratings_test = ratings_test.drop('timestamp')

    #cast movieId and userId to ints and rating to floats
    ratings_val = ratings_val.withColumn('movieId', ratings_val['movieId'].cast('integer'))
    ratings_val = ratings_val.withColumn('userId', ratings_val['userId'].cast('integer'))    
    ratings_val = ratings_val.withColumn('rating', ratings_val['rating'].cast('float')) 

    ratings_test = ratings_test.withColumn('movieId', ratings_test['movieId'].cast('integer'))
    ratings_test = ratings_test.withColumn('userId', ratings_test['userId'].cast('integer'))
    ratings_test = ratings_test.withColumn('rating', ratings_test['rating'].cast('float'))
    
    window_truth_val = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth_val))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    val_map = eval_ALS(truth_val, user_recs)
    print(val_map)

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