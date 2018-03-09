#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:58:57 2018

@author: mson
"""

import numpy as np
import neural_network as nn
from activations import relu, sigmoid 
from sklearn import datasets
from sklearn.model_selection import train_test_split

def trn_val_tst(X, y, trn_prop, val_prop, tst_prop):
    assert(0.99 <= trn_prop + val_prop + tst_prop <= 1.01)    
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                                  test_size = val_prop + tst_prop,
                                                  random_state = 1)
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn,
                                                  test_size = tst_prop,
                                                  random_state = 1)
    return X_trn, y_trn, X_val, y_val, X_tst, y_tst

def standardize_cols(X):
    return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

X = np.array([[1, 2, 3],
              [3, 2.5, 9],
              [8, 99, 4]])
y = np.array([0, 1, 0])
layer_dims = [3, 4, 2, 1]
net = nn.Net(layer_dims, [relu, relu, relu, sigmoid])
net.train(X, y, debug = True, iterations = 1, learning_rate=0.001)

## Something is happening that sends (layer n - 1).A to near zero. All negatives
## before??

iris = datasets.load_iris()
X, y = iris.data, iris.target
y_binary = np.array([1 if el in (1, 2) else 0 for el in y])
X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(X, y_binary, 2/3, 1/6, 1/6)
ilayer_dims = [X.shape[1], 4, 1]
iris_net = nn.Net(ilayer_dims, [relu, relu, sigmoid])

iris_net.train(standardize_cols(X_trn).T, y_trn, iterations = 5000, learning_rate = 0.01, 
               debug = True)
yhat_trn = iris_net.predict(standardize_cols(X_trn).T)
side_by_side = np.vstack((y_trn, yhat_trn)).T

