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
                          
    print('preds_truth rows:', preds_truth.count())
    
    eval_metrics = RankingMetrics(preds_truth)
    mean_avg_precision = eval_metrics.meanAveragePrecision
    
    return mean_avg_precision

def fit_eval_ALS(spark, ratings_train, truth_val):
    
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', 
              coldStartStrategy='drop', rank=200, maxIter=10, regParam=0.005)

    als_model = als.fit(ratings_train)
    
    preds_val = als_model.recommendForAllUsers(100)
    take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
    
    val_map = eval_ALS(truth_val, preds_val)
    
    return als_model, val_map

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
    window_truth_val = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth_val))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    als_model, full_als_map = fit_eval_ALS(spark, ratings_train, truth_val)
    
    # Get holdout set of movieIds, remove from training set, and train new ALS model (simulate cold start)
    item_factors = als_model.itemFactors.persist()
    movieIds = item_factors.select('id').persist()
    movieIds_held_out_df = movieIds.sample(fraction=0.1, seed=1)
    movieIds_held_out = np.array(movieIds_held_out_df.collect()).flatten()
    
    ratings_train_cold = ratings_train.filter(~col('movieId').isin(movieIds_held_out.tolist()))
    cold_ALS_model, cold_map = fit_eval_ALS(spark, ratings_train_cold, truth_val)
    
    # Get new model's user / item factors
    user_factors_cold = cold_ALS_model.userFactors.persist()
    item_factors_cold = cold_ALS_model.itemFactors.persist()
    
    user_factors_cold_matrix = np.array(user_factors_cold.select('features').rdd.map(lambda row: np.array(row['features'])).collect())
    item_factors_cold_matrix = np.array(item_factors_cold.select('features').rdd.map(lambda row: np.array(row['features'])).collect())
    
    # Train / validate content model to use genome data (features) to predict item factors
    tag_genome = spark.read.parquet(path_to_file + 'tag_genome_pivot.parquet', header='true')
    item_factors_train_val = item_factors_cold.join(tag_genome, item_factors_cold.id==tag_genome.movieId)\
                                        .drop('id')\
                                        .withColumnRenamed('features', 'target')
    alphas = np.array([0.1, 1, 10, 25, 50, 75, 100])
    content_model = train_content_regressor(spark, item_factors_train_val, alphas, path_to_file)   
    
    # Join held out movies to their tag genome data, then predict their item factors using content model
    item_factors_test = movieIds_held_out_df.join(tag_genome, movieIds_held_out_df.id==tag_genome.movieId)\
                                            .drop('id')
    X_ind = [str(i) for i in range(1,1129)]
    X_test = np.array(item_factors_test.select(X_ind).collect())
    held_out_factors_pred = content_model.predict(X_test)
    
    print('Creating combined items matrix..')
    # Combine item factors from ALS with item factors from content model
    item_factors_matrix_combined = np.vstack((item_factors_cold_matrix, held_out_factors_pred))
    print('Done')
    
    # Similarly, combine movieIds used in ALS with movieIds used in content model
    # Here, index of movie A in movieIds = index of movie A in item_factors_matrix_combined
    movieIds_cold = np.array(item_factors_cold.select('id').collect()).flatten()
    movieIds = np.concatenate((movieIds_cold, movieIds_held_out))
    
    
    # Evaluate cold start model
    print('Computing utility matrix..')
    userIds = np.array(user_factors_cold.select('id').collect()).flatten()
    utility_mat = user_factors_cold_matrix @ item_factors_matrix_combined.T
    print('Done\n')
    print('Utility Matrix shape:', utility_mat.shape)
    
    # user_recs_cold_start = []
    # user_recs_columns = ['userId', 'recommendations']
    # top_n = 100
    
    # for i in range(len(userIds)):
    #     row = utility_mat[i,:]
    #     user_id = userIds[i].tolist()
    #     ind = np.argpartition(row, -top_n)[-top_n:]
    #     ind = ind[np.argsort(row[ind])][::-1]
    #     top_n_movieIds = movieIds[ind].tolist()
    #     user_recs_cold_start.append((user_id, top_n_movieIds))
    
    # print(user_recs_cold_start[:10])
        
    # user_recs_df = spark.createDataFrame(data=user_recs_cold_start, schema=user_recs_columns)
    # user_recs_df.show(10)
    # truth_val.show(10)
    
    # cold_start_map = eval_ALS(truth_val, user_recs_df)
    
    # print('Full ALS MAP:', full_als_map)
    # print('ALS with content cold start on 10% movies:', cold_start_map)

 
# Only enter this block if we're in main
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
