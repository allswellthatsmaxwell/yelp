#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:42:03 2018

@author: mson
"""

class Layer:
    import activations as actv
    
    ## should probably replace W and b with shape, and have init make them itself!
    def __init__(self, name, W, b, activation):
        """ W: a m-by-n matrix
            b: an m-row column vector 
            activation: the activation function for this node
            name: the name of this node 
        """
        assert(W.shape[0] == b.shape[0])
        assert(b.shape[1] == 1)

        self.name = name
        self.W = W
        self.b = b
        self.A = None
        self.Z = None
    
    def shape(self): return W.shape
    def n_features(self): return self.shape[0]
    
    def propagate_forward_from(layer):
        """
        Performs forward propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        self.Z = np.dot(self.W, layer.A) + self.b
        
    def propagate_backward_from(layer):
        """
        Performs back propagation through this layer. 
        If this is layer n, then the layer argument is layer n + 1.
        """
        m = layer.A.shape[1]
        dZ = actv.derivative(self.activation)(layer.dA) ## should this be self.dA or layer.dA?
        self.dW = (1 / m) * np.dot(dZ, layer.A.T)
        self.db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        
    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class Net:
    """ A Net is made of layers
    """
    def __init__(self, layer_dims, activations):
        """
        layer_dims: an array of layer dimensions
        activations: an array of activation 
                     functions (each from the activations module); 
                     one function per layer
        """
        assert(len(layer_dims) == len(activations))
        self.layers = [Layer(layer_dims[i], activations[i]) for i in range(len(layer_dims))]
    
    
    def __assert_ok_topology(self, l_n, l_n_minus_1):
        l_n.shape[1] == l_n_minus_1.shape[0]
