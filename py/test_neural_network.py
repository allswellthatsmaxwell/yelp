#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:58:57 2018

@author: mson
"""

import numpy as np
import neural_network as nn
from activations import relu, sigmoid 

X = np.array([[1, 2, 3],
              [3, 2.5, 9],
              [8, 99, 4]])
y = np.array([0, 1, 0]).reshape((1, 3))


layer_dims = [3, 4, 2, 1]

net = nn.Net(layer_dims, [relu, relu, relu, sigmoid], 0.03)
net.train(X, y)

