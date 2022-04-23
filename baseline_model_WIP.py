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
    ratings_train = spark.read.csv(path_to_file + 'ratings_train.csv', header='true')
    ratings_val = spark.read.csv(path_to_file + 'ratings_val.csv', header='true')
    ratings_test = spark.read.csv(path_to_file + 'ratings_test.csv', header='true')
    
    ratings_train.createOrReplaceTempView('ratings_train')
    ratings_val.createOrReplaceTempView('ratings_val')
    ratings_test.createOrReplaceTempView('ratings_test')
    
    #create baseline ranking

    damping_factor = 0
    ratings_train = ratings_train.groupBy('movieId').agg(F.sum('rating').alias('rating_sum'), F.count('rating').alias('rating_count'))
    ratings_train = ratings_train.withColumn('rating_score', ratings_train.rating_sum / (ratings_train.rating_count + damping_factor))    
    
    ratings_train = ratings_train.toPandas()
    ratings_train = ratings_train.sort_values('rating_score', ascending=False).head(100)
    print('ratings_train baseline sorted DF:')
    print(ratings_train)
    
    print('sorted movies list:')
    print(ratings_train['movieId'].values)
    
    #create baseline rankings list from modified ratings_train dataframe
    #TO FIX: baseline ranking list does not currently preserve order from sorted ratings_train df                                 
    # baseline_ranking_list = ratings_train.select('movieId')#.rdd.flatMap(lambda x: x).collect()[:100]
    # print('baseline rankings by movieId:')
    # print(baseline_ranking_list)    
          
                   
    #create ground truth rankings by user from validation set
    
# =============================================================================
#     windowval = Window.partitionBy('userId').orderBy(F.col('rating').desc())
#     ratings_val = ratings_val.withColumn('rating_count', F.row_number().over(windowval))    
#     
#     ratings_val = ratings_val.filter(ratings_val.rating_count<=100)
# 
#     ratings_val  = ratings_val.groupBy('userId').agg(collect_list('movieId'))
#     
#     print('ground truth rankings by user from validation set:')
#     ratings_val.show()
#     
# 
#     #create rdd to imput into RankingMetrics evaluation
#     pred_and_labels = [(baseline_ranking_list, row['collect_list(movieId)']) for row in ratings_val.rdd.collect()]
#     pred_and_labels_rdd = spark.sparkContext.parallelize(pred_and_labels)    
#         
#     
#     #evaluate baseline rankings on validation set with rankingMetrics
#     eval_metrics = RankingMetrics(pred_and_labels_rdd)
#     print('mean average precision: ', eval_metrics.meanAveragePrecision)
#     
# =============================================================================

    
    
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
    
    
    
    
    
    
    
    
    
    
