#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:47:25 2018

@author: mson
"""

import numpy as np
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
