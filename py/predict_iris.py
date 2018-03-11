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
from sklearn.metrics import roc_auc_score
from prediction_utils import trn_val_tst
import loss_functions as losses

iris = datasets.load_iris()
X, y = iris.data, iris.target
y_binary = np.array([1 if el in (1, 2) else 0 for el in y])
X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(X, y_binary, 
                                                       4/10, 3/10, 3/10)
ilayer_dims = [X.shape[1], 4, 1]
iris_net = nn.Net(ilayer_dims, [relu, relu, sigmoid], loss = losses.LogLoss())

iris_net.train(X_trn.T, y_trn, iterations = 200, learning_rate = 0.05, 
               beta1 = 0.7, beta2 = 0.9,
               debug = True)
yhat_trn = iris_net.predict(X_trn.T)
yhat_val = iris_net.predict(X_val.T)
yyhat_trn = np.vstack((y_trn, yhat_trn)).T
yyhat_val = np.vstack((y_val, yhat_val)).T

auc_val = roc_auc_score(y_val, yhat_val)
print("AUC =", auc_val)