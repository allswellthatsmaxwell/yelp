#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:35:40 2018

@author: mson

initialization functions for initializing the W parameter in a single
Layer
"""

import numpy as np

def random_initialization(n, n_prev, scaling_factor):
    return scaling_factor * np.random.randn(n, n_prev)

def random(n, n_prev):
    return random_initialization(n, n_prev, 0.01)

def he(n, n_prev):
    return random_initialization(n, n_prev, np.sqrt(2 / n_prev))

def xavier(n, n_prev):
    return random_initialization(n, n_prev, np.sqrt(2 / n_prev))    
