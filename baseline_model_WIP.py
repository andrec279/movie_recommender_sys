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


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    #load train, val, test data into DataFrames
    ratings_train = spark.read.csv(f'hdfs:/user/{netID}/ratings_train.csv', 
				   header='true', 
				   schema='index INT, userId INT, movieID INT, rating FLOAT, timestamp INT, user_row_num INT, user_row_percentile FLOAT')
    ratings_val = spark.read.csv(f'hdfs:/user/{netID}/ratings_val.csv', 
				 header='true',
				 schema='index INT, userId INT, movieID INT, rating FLOAT, timestamp INT, user_row_num INT, user_row_percentile FLOAT')
    ratings_test = spark.read.csv(f'hdfs:/user/{netID}/ratings_test.csv', 
				  header='true',
				  schema='index INT, userId INT, movieID INT, rating FLOAT, timestamp INT, user_row_num INT, user_row_percentile FLOAT')
    
    ratings_train.createOrReplaceTempView('ratings_train')
    ratings_val.createOrReplaceTempView('ratings_val')
    ratings_test.createOrReplaceTempView('ratings_test')
    
    #create baseline ranking

    damping_factor = 0
    
    ratings_train = ratings_train.groupBy('movieId').agg(F.sum('rating').alias('rating_sum'), F.count('rating').alias('rating_count'))
    ratings_train.show()
    
    ratings_train = ratings_train.withColumn('rating_score', ratings_train.rating_sum / (ratings_train.rating_count + damping_factor))    
    ratings_train.show()

    ratings_train = ratings_train.sort('rating_score', ascending=False)
    ratings_train.show()

    #baseline_ranking  = ratings_train['movieId']

#    baseline_ranking = spark.sql('''
#                                 SELECT movieId
#                                 FROM(
#                                     SELECT movieId, AVG(rating)
#                                     FROM ratings_train
#                                     GROUP BY 1
#                                     ORDER BY 2 DESC
#                                     LIMIT 100
#				     ) as a
#                                 ''')
    
    #print('baseline rankings by movieId:')
    #baseline_ranking.show() 
    
    #convert baseline_ranking to list                             
    baseline_ranking_list = ratings_train.select('movieId').rdd.flatMap(lambda x: x).collect()
    print(baseline_ranking_list[:20])    
                             
    #create ground truth rankings by user from validation set
    windowval = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    ratings_val = ratings_val.withColumn('rating_count', F.row_number().over(windowval))    
    
    ratings_val = ratings_val.filter(ratings_val.rating_count<=100)

    ratings_val  = ratings_val.groupBy('userId').agg(collect_list('movieId'))
    
    print('ground truth rankings by user from validation set:')
    ratings_val.show()
    

    #create rdd to imput into RankingMetrics evaluation
    pred_and_labels = [(baseline_ranking_list, row['collect_list(movieId)']) for row in ratings_val.rdd.collect()]
    pred_and_labels_rdd = spark.sparkContext.parallelize(pred_and_labels)    
        
    
    #evaluate baseline rankings on validation set with rankingMetrics
    eval_metrics = RankingMetrics(pred_and_labels_rdd)
    print('mean average precision: ', eval_metrics.meanAveragePrecision)
    

    
    
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID) 
    
    
    
    
    
    
    
    
    
    
