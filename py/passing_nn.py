#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:42:03 2018

@author: mson
"""

class Layer:
    
    backward_map = {__relu: __relu_backward, 
                    __sigmoid: __sigmoid_backward}
    
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
    
    def __sigmoid(Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def __relu(Z):
        A = np.maximum(0, Z)
        return A
    
    def __relu_backward(dA):
        """
        Perform backward propagation for a single RELU unit.    
        Arguments: dA -- post-activation gradient, of any shape
        Returns: dZ -- Gradient of the cost with respect to Z
        """
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        assert (dZ.shape == self.Z.shape)
        # When z <= 0, you should set dz to 0 as well. 
        dZ[self.Z <= 0] = 0
        return dZ

    def __sigmoid_backward(dA):
        """
        Perform backward propagation for a single SIGMOID unit.
        Arguments: dA -- post-activation gradient, of any shape
        Returns: dZ -- Gradient of the cost with respect to Z
        """
        s = 1 / (1 + np.exp(-self.Z))
        dZ = dA * s * (1 - s)    
        assert (dZ.shape == self.Z.shape)    
        return dZ

    def __get_backward_activation(activation):
        backward_map[activation]
        
    def shape(self): return W.shape
    
    def propagate_forward_from(layer):
        """
        Performs forward propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        self.A = np.dot(self.W, layer.A) + self.b
        
    def propagate_backward_from(layer):
        """
        Performs back propagation through this layer. 
        If this is layer n, then the layer argument is layer n + 1.
        """
        
    def update_parameters(self, learning_rate):
    

class Net:
    """ A Net is made of layers
    """
    def __init__(self, layers):
        self.layers = layers
    