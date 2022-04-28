#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:23:37 2022

@author: jinishizuka
"""

#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_list
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import row_number, lit
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
    ratings_train = spark.read.csv(path_to_file + 'ratings_train{}.csv'.format(size), header='true')
    ratings_val = spark.read.csv(path_to_file + 'ratings_val{}.csv'.format(size), header='true')
    ratings_test = spark.read.csv(path_to_file + 'ratings_test{}.csv'.format(size), header='true')
    
    ratings_train.createOrReplaceTempView('ratings_train')
    ratings_val.createOrReplaceTempView('ratings_val')
    ratings_test.createOrReplaceTempView('ratings_test')
          
                   
    #create ground truth rankings by user from validation set and test set
    windowval = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    
    ratings_val = ratings_val.withColumn('rating_count', F.row_number().over(windowval))    
    ratings_val = ratings_val.filter(ratings_val.rating_count<=100)
    ratings_val  = ratings_val.groupBy('userId').agg(collect_list('movieId'))
    
    ratings_test = ratings_test.withColumn('rating_count', F.row_number().over(windowval))
    ratings_test = ratings_test.filter(ratings_test.rating_count<=100)
    ratings_test = ratings_test.groupby('userId').agg(collect_list('movieId'))
    
    #sort ratings_train df by movie rating score with various damping factors
    if full_data == True:
        damping_factors = [100]
    else:
        damping_factors = [1, 5, 10, 50, 100]
    map_results = np.empty(len(damping_factors))
    for i in range(len(damping_factors)):
        ratings_train_iter = ratings_train.groupBy('movieId').agg(F.sum('rating').alias('rating_sum'), F.count('rating').alias('rating_count'))
        ratings_train_iter = ratings_train_iter.withColumn('rating_score', ratings_train_iter.rating_sum / (ratings_train_iter.rating_count + damping_factors[i]))    
        ratings_train_iter = ratings_train_iter.sort('rating_score', ascending=False)

        #create baseline rankings list                               
        ratings_train_pd = ratings_train_iter.toPandas()
        baseline_ranking_list = list(ratings_train_pd['movieId'])[:100]
    
        #create rdd to imput into RankingMetrics evaluation
        pred_and_labels = [(baseline_ranking_list, row['collect_list(movieId)']) for row in ratings_val.rdd.collect()]
        pred_and_labels_rdd = spark.sparkContext.parallelize(pred_and_labels)    
     
        #evaluate baseline rankings on validation set with rankingMetrics
        eval_metrics = RankingMetrics(pred_and_labels_rdd)
        map_results[i] = eval_metrics.meanAveragePrecision
        
    best_damping_factor = damping_factors[np.argmax(map_results)]
    print('Best mean average precision from validation set: ', np.max(map_results))
    print('Best damping factor from validation set: ', best_damping_factor)
    
    # Evaluate best damping factor on Test
    pred_and_labels = [(baseline_ranking_list, row['collect_list(movieId)']) for row in ratings_test.rdd.collect()]
    pred_and_labels_rdd = spark.sparkContext.parallelize(pred_and_labels)
    eval_metrics = RankingMetrics(pred_and_labels_rdd)
    print('Final baseline model mean average precision on test set: ', eval_metrics.meanAveragePrecision)
    
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    local_source = False # For local testing
    full_data = True
    
    size = '-small' if full_data == False else ''
    
    print('NOTE: Local mode set to {} and size is set to {}'.format(local_source, size))
     
    if local_source == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 
    
    
    
    
    
    
    
    
    
    
