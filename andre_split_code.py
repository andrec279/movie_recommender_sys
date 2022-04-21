#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:04:58 2022

@author: andrechen
"""

import pandas as pd
import time
import numpy as np

full_data = True

size = '-small' if full_data==False else ''

files = {'ratings': 'ml-latest{}/ratings.csv'.format(size),
         'movies': 'ml-latest{}/movies.csv'.format(size),
         'links': 'ml-latest{}/links.csv'.format(size),
         'tags': 'ml-latest{}/tags.csv'.format(size)}

if full_data == True:
    files['genome-scores'] = 'ml-latest{}/genome-scores.csv'.format(size)
    files['genome-tags'] = 'ml-latest{}/genome-tags.csv'.format(size)

#%%
'''(1) PANDAS WORKFLOW'''

'''

Memory Usage (GB) in Pandas across all 6 DataFrames (full dataset):
========================
ratings: 0.888110336GB
movies: 0.00139448GB
links: 0.00139448GB
tags: 0.035488032GB
genome-scores: 0.3567008GB
genome-tags: 1.8176e-05GB
========================
Total:  1.28 Gigabytes

'''

df_ratings = pd.read_csv(files['ratings'], header=0)
print('Total Users:', len(pd.unique(df_ratings['userId'])))

'''
Total Users: 283228
'''

#%%

# Get indices for each user and subsample (runs in about 3.5 minutes)

'''
60/20/20 Train/Val/Test Split on individual user basis using df.groupby.sample()

** Takes ~3.5 minutes to run

TODO: Incorporate time-based splitting - train on earlier examples, predict on later examples
'''

t0 = time.time()

# 1. Train split: Sample 60% of interactions for each user
ratings_train = df_ratings.groupby('userId').sample(frac=0.5, replace=False)
train_index = ratings_train.index.to_list()

# 2. Validation split: Filter out training indices from original data, 
# sample 50% of remaining interactions for each user
ratings_val = df_ratings.loc[df_ratings.index.difference(train_index), :].groupby('userId').sample(frac=0.5, replace=False)
train_val_index = train_index + ratings_val.index.to_list()

# 3. Test split: Filter training and validation indices out of full dataset
ratings_test = df_ratings.loc[df_ratings.index.difference(train_val_index), :]

# 4. Remove individuals from val / test with fewer than [threshold] ratings
threshold = 5
ratings_val = ratings_val.groupby('userId').filter(lambda x: len(x) > threshold)
ratings_test = ratings_test.groupby('userId').filter(lambda x: len(x) > threshold)

print('Pandas workflow takes {} minutes'.format((time.time() - t0)/60))

#%%

'''
Run the following filters and checks to ensure that:
    
1. All users in the training set appear in the validation and test set 
   (Note: There is no overlap in interactions data between the three splits)
   
2. All movies in the training set appear in the validation and test set 

3. There are enough ratings per person (at least 4?) in all three splits

4. There is enough data overall in all three splits
'''

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


'''
Results:
    
====================
Before Filtering:
====================

Total DF len: 27021197
Train DF len: 13508246
Val DF len: 6750827
Test DF len: 6762124

Number of users in each split:
Total: 202390
Train: 202390
Val: 202390
Test: 202390

Number of movies in each split:
Total: 53812
Train: 46404
Val: 38687
Test: 38846

====================
After Filtering:
====================

Total DF len: 27021197
Train DF len: 13508246
Val DF len: 6745369
Test DF len: 6756618

Number of users in each split:
Total: 202390
Train: 202390
Val: 202390
Test: 202390

Number of movies in each split:
Total: 53812
Train: 46404
Val: 34311
Test: 34426

Minimum Ratings per User:
Train: 8
Validation: 4
Test: 4
'''

#%%

ratings_train.to_csv('ratings_train.csv')
ratings_val.to_csv('ratings_val.csv')
ratings_test.to_csv('ratings_test.csv')
