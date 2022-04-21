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
Total Users: 283228
Users with more than 10 ratings: 236658

'''

# Filter to users that have rated at least [threshold] movies to have enough data in each split
threshold = 10
df_ratings = pd.read_csv(files['ratings'], header=0)
print('Total Users:', len(pd.unique(df_ratings['userId'])))
df_ratings = df_ratings.groupby('userId').filter(lambda x: len(x) > threshold)
print(f'Users with more than {threshold} ratings:', len(pd.unique(df_ratings['userId'])))

#%%

# Get indices for each user and subsample (runs in about 3.5 minutes)

'''
60/20/20 Train/Val/Test Split

1. Train split: Sample 60% of interactions for each user
2. Validation split: Filter training indices out of full dataset, 
   sample 50% of remainder
3. Test split: Filter training and validation indices out of full dataset

'''

t0 = time.time()

ratings_train = df_ratings.groupby('userId').sample(frac=0.5, replace=False)
train_index = ratings_train.index.to_list()

ratings_val = df_ratings.loc[df_ratings.index.difference(train_index), :].groupby('userId').sample(frac=0.5, replace=False)
train_val_index = train_index + ratings_val.index.to_list()

ratings_test = df_ratings.loc[df_ratings.index.difference(train_val_index), :]

print('Pandas workflow takes {} minutes'.format((time.time() - t0)/60))

#%%

print('''====================
Before Splitting:
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

'''
Find rows corresponding movies in val and test splits that are 
NOT in train split and remove
'''

val_missing_from_train = [movieId for movieId in val_movie_ids if movieId not in train_movie_ids]
test_missing_from_train = [movieId for movieId in test_movie_ids if movieId not in train_movie_ids]

ratings_val_filtered = ratings_val[~ratings_val['movieId'].isin(val_missing_from_train)]
ratings_test_filtered = ratings_test[~ratings_test['movieId'].isin(test_missing_from_train)]


print('''\n====================
After Splitting:
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
Before Splitting:
====================

Total DF len: 27495774
Train DF len: 13755953
Val DF len: 6874893
Test DF len: 6864928

Number of users in each split:
Total: 236658
Train: 236658
Val: 236658
Test: 236658

Number of movies in each split:
Total: 53848
Train: 46453
Val: 38895
Test: 38943

====================
After Splitting:
====================

Total DF len: 27495774
Train DF len: 13755953
Val DF len: 6869443
Test DF len: 6859441

Number of users in each split:
Total: 236658
Train: 236658
Val: 236658
Test: 236658

Number of movies in each split:
Total: 53848
Train: 46453
Val: 34506
Test: 34519

Minimum Ratings per User:
Train: 6
Validation: 2
Test: 3
'''

#%%

ratings_train.to_csv('ratings_train.csv')
ratings_val.to_csv('ratings_val.csv')
ratings_test.to_csv('ratings_test.csv')
