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


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    #load train, val, test data into DataFrames
    ratings_train = spark.read.csv(f'hdfs:/user/{netID}/ratings_train.csv')
    ratings_val = spark.read.csv(f'hdfs:/user/{netID}/ratings_val.csv')
    ratings_test = spark.read.csv(f'hdfs:/user/{netID}/ratings_test.csv')
    
    ratings_train.createOrReplaceTempView('ratings_train')
    ratings_val.createOrReplaceTempView('ratings_val')
    ratings_test.createOrReplaceTempView('ratings_test')
    
    #create baseline ranking
    baseline_ranking = spark.sql('''
                                 SELECT movieId
                                 FROM
                                     SELECT movieId, AVG(rating)
                                     FROM ratings_train
                                     GROUP BY 1
                                     ORDER BY 2 DESC
                                     LIMIT 100
                                 ''') 
                                 
    baseline_ranking_list = baseline_ranking.select('movieId').rdd.flatMap(lambda x: x).collect()
                                 
    #create ground truth rankings by user from validation set
    #to do: limit top 100 movies by user
    ratings_val = ratings_val.sort_values(by=['userId', 'rating'], ascending=False)
    
    #create rdd to imput into RankingMetrics evaluation
    eval_df = ratings_val.groupBy('userId').agg(collect_list('movieId'))
    eval_df = eval_df.withColumn('true_pred_pair',  (baseline_ranking_list, eval_df['movieID']))
    
    eval_df.show()
    
    
    
    
    
    
    
    
    
    
    
    
    