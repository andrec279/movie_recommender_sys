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
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from functools import reduce
import numpy as np
import time
import random

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
        
    t0 = time.time()
    schema = 'userId INT, movieId INT, rating FLOAT, timestamp INT'
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

    # Get the predicted rank-ordered list of movieIds for each user
    window_truth_val = Window.partitionBy('userId').orderBy(F.col('rating').desc())
    truth_val = ratings_val.withColumn('rating_count', F.row_number().over(window_truth_val))
    truth_val = truth_val.filter(truth_val.rating_count<=100)
    truth_val = truth_val.groupby('userId').agg(collect_list('movieId').alias('true_ranking'))
    
    t_prep = time.time()
    print('Data preparation time (.parquet): ', round(t_prep-t0, 3), 'seconds')
    
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', 
              coldStartStrategy='drop', rank=10, maxIter=5, regParam=0.005)

    als_model = als.fit(ratings_train)
    
    # Get user and item factors
    user_factors = als_model.userFactors
    item_factors = als_model.itemFactors
    
    tag_genome = spark.read.parquet(path_to_file + 'tag_genome_pivot.parquet', header='true')
    item_factors_features = item_factors.join(tag_genome, item_factors.id==tag_genome.movieId)\
                                        .drop('id')\
                                        .withColumnRenamed('features', 'target')
    
    # For spark models, need to vectorize feature columns into one column
    vectorAssembler = VectorAssembler(inputCols = list(range(1, 1129)), outputCol = 'features')
    item_factors_features_v = vectorAssembler.transform(item_factors_features)
    item_factors_features_v.show(3)
    
    splits = item_factors_features_v.randomSplit([0.7, 0.3])
    train_features = splits[0]
    val_features = splits[1]
    
    # Train model
    maxIters = [5, 10, 15]
    regParams = [1e-4, 1e-3, 1e-2, 1e-1]
    lr = LinearRegression(featuresCol = 'features', labelCol='target')
    param_grid = ParamGridBuilder()\
                    .addGrid(als.maxIter, maxIters)\
                    .addGrid(als.regParam, regParams)\
                    .build()
    evaluatorRMSE = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

    CV_lr = CrossValidator(estimator=lr,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluatorRMSE,
                           numFolds=5)
    
    CV_lr_fitted = CV_lr.fit(train_features)
    lr_best = CV_lr_fitted.bestModel
    trainingSummary = lr_best.summary
    print('Training RMSE: {}'.format(trainingSummary.rootMeanSquaredError))
    print('Training R^2: {}'.format(trainingSummary.r2))
    
    val_feature_pred = lr_best.transform(val_features)
    evaluatorR2 = RegressionEvaluator(predictionCol='prediction', labelCol='target', metricName='r2')
    
    print('Validation RMSE: {}'.format(evaluatorRMSE.evaluate(val_feature_pred)))
    print('Validation R^2: {}'.format(evaluatorR2.evaluate(val_feature_pred)))
    
    
    'TO-DO: evaluation section of full content cold start model'
    
    # # each set of user recommendations is in format [[movieId1, rating_pred1], [movieId2, rating_pred2], ...]
    # # we can drop ratings and restructure to [movieId1, movieId2, ...]
    # take_movieId = udf(lambda rows: [row[0] for row in rows], ArrayType(IntegerType()))
    # preds_val = preds_val.withColumn('recommendations', take_movieId('recommendations'))
    
    # # Join truth and predictions and evaluate
    # preds_truth = truth_val.join(preds_val, truth_val.userId == preds_val.userId, 'inner')\
    #                       .select(col('true_ranking'), col('recommendations'))\
    #                       .rdd

    # eval_metrics = RankingMetrics(preds_truth)
    # val_map = eval_metrics.meanAveragePrecision
    # print('Validation MAP after model tuning: ', val_map)
    
    # t_complete = time.time()
    # print('\nTraining time (.csv) for {} configurations: {} seconds'.format(len(ranks)*len(regParams)*len(maxIters), round(t_complete-t_prep,3)))
    # print('\nTotal runtime: {} seconds'.format(round(t_complete-t0, 3)))

 
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    local_source = False # For local testing
    full_data = False
    size = '-small' if full_data == False else ''
     
    if local_source == False:
        # Get user netID from the command line
        netID = getpass.getuser()
    else:
        netID = None

    # Call our main routine
    main(spark, netID) 
