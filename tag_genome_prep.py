#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 21:12:55 2022

@author: andrechen
"""

#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def main(spark, netID=None):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    
    if local_source == False:
        filepath = f'hdfs:/user/{netID}/'
    else:
        filepath = ''
    
    schema = 'movieId INT, tagId INT, relevance FLOAT'
    tag_genome = spark.read.csv(filepath + 'genome-scores.csv', schema=schema)
    tag_genome_pivot = tag_genome.groupBy('movieId').pivot('tagId').sum('relevance').drop(col('null'))
    print(tag_genome_pivot.count())
    tag_genome_pivot.write.mode('overwrite').option('header', True).parquet(filepath + 'tag_genome_pivot.parquet')

    print('\nFiles written successfully to {}'.format('local folder' if local_source == True else filepath))


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