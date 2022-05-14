#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cell 0
"""
Created on Wed Apr 20 14:04:58 2022

@author: andrechen
"""

import pandas as pd
import time
import numpy as np
import getpass
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id, col
import random
import math

def main(spark, netID):
    t0 = time.time()
    if local_save==True:
        path = 'ml-latest{}/'.format(size)
    else:
        path = f'hdfs:/user/{netID}/'

    files = {'ratings': path + 'ratings.csv',
             'movies': path + 'movies.csv',
             'links': path + 'links.csv',
             'tags': path + 'tags.csv'}
    
    if full_data == True:
        files['genome-scores'] = path + 'genome-scores.csv'
        files['genome-tags'] = path + 'genome-tags.csv'
        
    # Read in and add temporary row_id column for splitting
    df_ratings = spark.read.csv(files['ratings'], header='true')
    
    # Add percentile column, ordered by timestamp and row_id column for splitting
    user_window_time = Window.partitionBy('userId').orderBy(F.col('timestamp'))
    df_ratings = df_ratings.withColumn('row_id', monotonically_increasing_id())
    df_ratings = df_ratings.withColumn('time_pct', F.percent_rank().over(user_window_time))
    
    # define function for random sampling from each userId
    def sample(iter, frac): 
        list_iter = list(iter)
        n = math.ceil(len(list_iter)*frac)
        rs = random.Random()
        return rs.sample(list(iter), n)  
    
    '''SPLITTING SECTION'''
    # 60/20/20 Train/Val/Test Split on individual user basis
    print('Splitting into train / val /test...')
    
    # Get 40% earliest interactions and 20% latest interactions per user for training set
    train_lower_frac = 0.4
    train_upper_frac = 0.8
    
    # Get training split
    ratings_train = df_ratings.filter((df_ratings['time_pct']<=train_lower_frac)|(df_ratings['time_pct']>train_upper_frac))\
                              .persist()
    
    # Filter out ratings_train from df_ratings
    # Remaining 40% of each user's data split evenly between val and test
    ratings_val_test = df_ratings.join(ratings_train, 'row_id', 'left_anti')
    
    
    # Sample 50% of remainder (ratings_val_test) to get validation split
    val_frac = 0.5
    user_groups = ratings_val_test.rdd.map(lambda row: (row.userId, row)).groupByKey()
    ratings_val = spark.createDataFrame(user_groups.flatMap(lambda r: sample(r[1], val_frac))).persist()

    # Remove validation split from ratings_val_test, remainder is the test split
    ratings_test = ratings_val_test.join(ratings_val, 'row_id', 'left_anti')
    
    # Remove row_id / timestamp pctile columns, we only needed them for splitting
    ratings_train = ratings_train.select('userId', 'movieId', 'rating', 'timestamp')
    ratings_val = ratings_val.select('userId', 'movieId', 'rating', 'timestamp')
    ratings_test = ratings_test.select('userId', 'movieId', 'rating', 'timestamp')
    print('Splitting done')

    '''FILTERING SECTION'''    
    # Filter in validation / test to only users with 5+ reviews
    threshold = 5
    print('Filtering validation and test to users with 5+ reviews and only movies in training set...')
    
    windowSpecCount = Window.partitionBy('userId')
    ratings_val = ratings_val.withColumn('movieCt', F.count(col('movieId')).over(windowSpecCount))
    ratings_test = ratings_test.withColumn('movieCt', F.count(col('movieId')).over(windowSpecCount))
    
    ratings_val = ratings_val.filter(ratings_val['movieCt']>=threshold)
    ratings_test = ratings_test.filter(ratings_test['movieCt']>=threshold)
    
    
    # Filter validation / test to only movies that are in training set
    train_movies = set(ratings_train.select('movieId').rdd.map(lambda row: row['movieId']).collect())
    ratings_val = ratings_val.filter(col('movieId').isin(train_movies))
    ratings_test = ratings_test.filter(col('movieId').isin(train_movies))
    
    print('Filtering done')
    
    # '---FOR DEBUGGING---'
    
    # print('\nSplit sizes:')
    # total_len = df_ratings.count()
    # train_count = ratings_train.count()
    # val_count = ratings_val.count()
    # test_count = ratings_test.count()
    # print('Training COUNT: {} Training PCT: {}'.format(train_count, train_count/total_len))
    # print('Validation COUNT: {} Validation PCT: {}'.format(val_count, val_count/total_len))
    # print('Test COUNT: {} Test PCT: {}'.format(test_count, test_count/total_len))
    
    # print('\nUsers in each split:')
    # print('Training: ', ratings_train.select('userId').distinct().count())
    # print('Validation: ', ratings_val.select('userId').distinct().count())
    # print('Test: ', ratings_test.select('userId').distinct().count())
    
    # print('\nMovies in each split:')
    # print('Training: ', ratings_train.select('movieId').distinct().count())
    # print('Validation: ', ratings_val.select('movieId').distinct().count())
    # print('Test: ', ratings_test.select('movieId').distinct().count())
    
    # '---FOR DEBUGGING---'
    
    if local_save == True:
        filepath = ''
    
    else:
        filepath = f'hdfs:/user/{netID}/'
    
    ratings_train.write.mode('overwrite').option('header', True).parquet(filepath + f'ratings_train{size}.parquet')
    ratings_val.write.mode('overwrite').option('header', True).parquet(filepath + f'ratings_val{size}.parquet')
    ratings_test.write.mode('overwrite').option('header', True).parquet(filepath + f'ratings_test{size}.parquet')

    print('\nFiles written successfully to {}'.format('local folder' if local_save == True else filepath))
    print('\nTest output for ratings_train:')
    print(spark.read.parquet(filepath + f'ratings_train{size}.parquet', header='true').show(10))
    
    print('Full runtime: ', time.time()-t0, 'seconds')
    
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    local_save = True # Set = True for debugging on local machine
    
    full_data = False
    size = '-small' if full_data==False else ''
    
    if local_save == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 
