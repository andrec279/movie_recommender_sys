#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:02:05 2022

@author: jinishizuka
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(file):
    data = np.genfromtxt(file)   
    
    train, test_val = train_test_split(data, train_size=0.8)
    
    num_cols = data.shape[1]
    sample_size = int(num_cols * 0.8)
    train_partial = np.empty(test_val.shape)
    
    for i in range(test_val.shape[0]):
        rand_indexes = np.random(range(num_cols), size=sample_size, replace=False)
        
        for j in rand_indexes:
            train_partial[i,j] = test_val[i,j]
            test_val[i,j] = np.NaN
    
    train = np.concat(train, train_partial)
    val, test = train_test_split(test_val, train_size=0.5)
    
    np.savetxt('training_data.csv', train, delimiter=',')
    np.savetxt('validation_data.csv', val, delimiter=',')
    np.savetxt('testing_data.csv', test, delimiter=',')
    
