#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:42:03 2018

@author: mson
"""

class Layer:
    import activations as actv
    
    def __init__(self, name, W, b, activation, lprev = None, lnext = None):
        """ W: a m-by-n matrix
            b: an m-row column vector 
            activation: the activation function for this node
            name: the name of this node 
            lprev: the layer previous to this layer
            lnext: the layer subsequent to this layer
        """
        self.name = name
        self.W = W
        self.b = b
        self.lnext = lnext
        self.lprev = lprev
        self.A = None
        self.Z = None
    
    def __assert_ok_topology(self, lprev):
        self.shape[1] == lprev.shape[0]
        self.W.shape[0] == b.shape[0]
        self.b.shape[1] == 1

    def shape(self): return W.shape
    def n_features(self): return self.shape[0]
    
    def propagate_forward_from(layer):
        """
        Performs forward propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        self.Z = np.dot(self.W, layer.A) + self.b
        
    def propagate_backward_to(layer):
        """
        Performs back propagation through this layer. 
        If this is layer n, then the layer argument is layer n + 1.
        """
        m = layer.A.shape[1]
        dZ = actv.derivative(self.activation)(lnext.dA)
        dW = (1 / m) * np.dot(dZ, layer.A.T)
        db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        
    def update_parameters(self, learning_rate):
    

class Net:
    """ A Net is made of layers
    """
    def __init__(self, layers):
        self.layers = layers
    