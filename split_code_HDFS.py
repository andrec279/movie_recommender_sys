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
from pyspark.sql import SparkSession

def main(spark, netID):

    if local_save==True:
        path = 'ml-latest{}/'.format(size)
    else:
        path = '/scratch/work/courses/DSGA1004-2021/movielens/ml-latest{}/'.format(size)

    files = {'ratings': path + 'ratings.csv',
             'movies': path + 'movies.csv',
             'links': path + 'links.csv',
             'tags': path + 'tags.csv'}
    
    if full_data == True:
        files['genome-scores'] = path + 'genome-scores.csv'
        files['genome-tags'] = path + 'genome-tags.csv'

    df_ratings = pd.read_csv(files['ratings'], header=0)
    print('Total Users:', len(pd.unique(df_ratings['userId'])))

    # Sort all ratings data by timestamp for each user
    df_ratings_timesort = df_ratings.sort_values(by=['userId', 'timestamp']).reset_index(drop=True)
    df_ratings_timesort['user_row_num'] = df_ratings_timesort.groupby('userId').cumcount()+1
    df_ratings_timesort['user_row_percentile'] = df_ratings_timesort['user_row_num']\
                / df_ratings_timesort.groupby('userId')['user_row_num'].transform('last')

    # 60/20/20 Train/Val/Test Split on individual user basis using df.groupby.sample()
    # NOTE: Data is split such that earlier interactions are in training data, later interactions
    # are split randomly between validation and test data
    
    t0 = time.time()
    ratings_train = df_ratings_timesort[df_ratings_timesort['user_row_percentile']<=0.6]
    ratings_val_test = df_ratings_timesort[df_ratings_timesort['user_row_percentile']>0.6]
    
    ratings_val = ratings_val_test.groupby('userId').sample(frac=0.5, replace=False, random_state=1)
    ratings_val_index = ratings_val.index.to_list()
    
    ratings_test = ratings_val_test.loc[ratings_val_test.index.difference(ratings_val_index), :]
    
    threshold = 5
    ratings_val = ratings_val.groupby('userId').filter(lambda x: len(x) >= threshold)
    ratings_test = ratings_test.groupby('userId').filter(lambda x: len(x) >= threshold)

    # Checks

    # Run the following filters and checks to ensure that:
    # 1. All users in the training set appear in the validation and test set 
    #    (Note: There is no overlap in interactions data between the three splits)
    # 2. All movies in the training set appear in the validation and test set 
    # 3. There are enough ratings per person (at least 4?) in all three splits
    # 4. There is enough data overall in all three splits


    print('''====================
    Before Filtering:
    ====================\n''')
    
    print('Total DF len:', len(df_ratings.index))
    print('Train DF len:', len(ratings_train.index))
    print('Val DF len:', len(ratings_val.index))
    print('Test DF len:', len(ratings_test.index))
    
    print('\nNumber of users in each split:')
    print('Total:', len(pd.unique(df_ratings['userId'])))
    print('Train:', len(pd.unique(ratings_train['userId'])))
    print('Val:', len(pd.unique(ratings_val['userId'])))
    print('Test:', len(pd.unique(ratings_test['userId'])))
    
    all_movie_ids = pd.unique(df_ratings['movieId'])
    train_movie_ids = pd.unique(ratings_train['movieId'])
    val_movie_ids = pd.unique(ratings_val['movieId'])
    test_movie_ids = pd.unique(ratings_test['movieId'])
    
    print('\nNumber of movies in each split:')
    print('Total:', len(all_movie_ids))
    print('Train:', len(train_movie_ids))
    print('Val:', len(val_movie_ids))
    print('Test:', len(test_movie_ids))
    
    
    # Find rows corresponding movies in val and test splits that are 
    # NOT in train split and remove
    
    val_missing_from_train = [movieId for movieId in val_movie_ids if movieId not in train_movie_ids]
    test_missing_from_train = [movieId for movieId in test_movie_ids if movieId not in train_movie_ids]
    
    ratings_val_filtered = ratings_val[~ratings_val['movieId'].isin(val_missing_from_train)]
    ratings_test_filtered = ratings_test[~ratings_test['movieId'].isin(test_missing_from_train)]
    
    
    print('''\n====================
    After Filtering:
    ====================\n''')
    
    print('Total DF len:', len(df_ratings.index))
    print('Train DF len:', len(ratings_train.index))
    print('Val DF len:', len(ratings_val_filtered.index))
    print('Test DF len:', len(ratings_test_filtered.index))
    
    print('\nNumber of users in each split:')
    print('Total:', len(pd.unique(df_ratings['userId'])))
    print('Train:', len(pd.unique(ratings_train['userId'])))
    print('Val:', len(pd.unique(ratings_val_filtered['userId'])))
    print('Test:', len(pd.unique(ratings_test_filtered['userId'])))
    
    print('\nNumber of movies in each split:')
    print('Total:', len(pd.unique(df_ratings['movieId'])))
    print('Train:', len(pd.unique(ratings_train['movieId'])))
    print('Val:', len(pd.unique(ratings_val_filtered['movieId'])))
    print('Test:', len(pd.unique(ratings_test_filtered['movieId'])))
    
    print('\nMinimum Ratings per User:')
    print('Train:', ratings_train.groupby('userId')['movieId'].count().min())
    print('Validation:', ratings_val.groupby('userId')['movieId'].count().min())
    print('Test:', ratings_test.groupby('userId')['movieId'].count().min())

    # Convert to spark to write to HDFS
    cols = ['userId', 'movieId', 'rating', 'timestamp']
    
    if local_save == True:
        ratings_train[cols].to_csv('ratings_train{}.csv'.format(size))
        ratings_val[cols].to_csv('ratings_val{}.csv'.format(size))
        ratings_test[cols].to_csv('ratings_test{}.csv'.format(size))
        
        print('\nTrain/Validation/Split files written successfully to local folder')
        print('\nTest output for ratings_train:')
        print(pd.read_csv(f'ratings_train{size}.csv').head(10))
    
    else:
    
        ratings_train = spark.createDataFrame(ratings_train[cols])
        ratings_val = spark.createDataFrame(ratings_val[cols])
        ratings_test = spark.createDataFrame(ratings_test[cols])
        
        ratings_train.write.mode('overwrite').option("header",True).csv(f'hdfs:/user/{netID}/ratings_train{size}.csv')
        ratings_val.write.mode('overwrite').option("header",True).csv(f'hdfs:/user/{netID}/ratings_val{size}.csv')
        ratings_test.write.mode('overwrite').option("header",True).csv(f'hdfs:/user/{netID}/ratings_test{size}.csv')
    
        print('\nTrain/Validation/Split files written successfully to HDFS')
        print('\nTest output for ratings_train:')
        print(spark.read.csv(f'hdfs:/user/{netID}/ratings_train{size}.csv', header='true').show(10))
    

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    local_save = False # Set = True for debugging on local machine
    
    full_data = True
    size = '-small' if full_data==False else ''
    
    if local_save == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 
