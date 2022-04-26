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
import numpy as np


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

    #load train, val, test data into DataFrames
    schema = 'index INT, userId INT, movieId INT, rating FLOAT, timestamp INT'
    ratings_train = spark.read.csv(path_to_file + 'ratings_train.csv', header='true', schema=schema)
    ratings_val = spark.read.csv(path_to_file + 'ratings_val.csv', header='true', schema=schema)
    ratings_test = spark.read.csv(path_to_file + 'ratings_test.csv', header='true', schema=schema)
    
    #create ground truth rankings by user from validation set and test set
    windowval = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    
    ratings_val = ratings_val.withColumn('rating_count', F.row_number().over(windowval))    
    ratings_val = ratings_val.filter(ratings_val.rating_count<=100)
    ratings_val  = ratings_val.groupBy('userId').agg(collect_list('movieId'))
    
    ratings_test = ratings_test.withColumn('rating_count', F.row_number().over(windowval))
    ratings_test = ratings_test.filter(ratings_test.rating_count<=100)
    ratings_test = ratings_test.groupby('userId').agg(collect_list('movieId'))
    
    # Fit ALS model
    regParams = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    val_map_scores = np.empty(len(regParams))
    
    for i in range(len(regParams)):
    
        als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', 
                  maxIter=10, rank=50, regParam=regParams[i])
        als_fitted = als.fit(ratings_train)
        
        # Get top 100 recommendations for each user
        userRecs = als_fitted.recommendForAllUsers(100)
        
        # One value in DF recommendations column is structured as [[movieId1, rating_pred1], [movieId2, rating_pred2], ...]
        # take_movieId gets value in the format [movieId1, movieId2, ...] while preserving order
        take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
        userRecs = userRecs.withColumn('recommendations', take_movieId('recommendations'))
        
        # Evaluate on validation set
        val_pred = ratings_val.join(userRecs, ratings_val.userId == userRecs.userId, 'inner')\
                              .select(col('collect_list(movieId)'), col('recommendations'))\
                              .rdd
    
        #val_pred_rdd = spark.sparkContext.parallelize(val_pred)
        eval_metrics = RankingMetrics(val_pred)
        val_map_scores[i] = eval_metrics.meanAveragePrecision
    
    print('Regularization constants: ', regParams)
    print('Associated Validation MAP Scores: ', val_map_scores)
    print('Best Regularization Parameter: ', regParams[np.argmax(val_map_scores)])
    print('Best Validation MAP Score: ', np.max(val_map_scores))
        
            
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