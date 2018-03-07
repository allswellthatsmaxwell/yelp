#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:42:03 2018

@author: mson
"""

import activations as actv
import initializations
import numpy as np

class InputLayer:
    
    def __init__(self, n):
        self.A = None

class Layer:

    def __init__(self, name, n, n_prev, activation, 
                 initialization = initializations.random):
        """ n: integer; the dimension of this layer
            n_prev: integer; the dimension of the previous layer            
            activation: function; the activation function for this node
            name: the name of this node 
            initialization: function; the initialization strategy to use
        """
        self.activation = activation
        self.W = initialization(n, n_prev)
        self.b = np.zeros((n, 1))        
        self.name = name        
        self.A = None
        self.Z = None
    
    def shape(self): return W.shape
    def n_features(self): return self.shape[0]
    
    def propagate_forward_from(self, layer):
        """
        Performs forward propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        self.Z = np.dot(self.W, layer.A) + self.b
        self.A = self.activation(self.Z)
        
    def propagate_backward_to(self, layer):
        """
        Performs back propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        m = layer.A.shape[1]
        dZ = actv.derivative(self.activation)(self.dA, self.Z) ## should this be self.dA or layer.dA?
        self.dW = (1 / m) * np.dot(dZ, layer.A.T)
        self.db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        layer.dA = np.dot(self.W.T, dZ) ## this is uncomfortable, design-wise
        
    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class Net:
    """ A Net is made of layers
    """
    def __init__(self, layer_dims, activations, learning_rate):
        """
        layer_dims: an array of layer dimensions. 
                    including the input layer.
        activations: an array of activation 
                     functions (each from the activations module); 
                     one function per layer
        """
        assert(len(layer_dims) == len(activations))
        
        self.learning_rate = learning_rate
        self.is_trained = False
        input_layer = InputLayer(layer_dims[0])
        self.layers = [input_layer]
        for i in range(1, len(layer_dims)):
            self.layers.append(
                Layer(name = i,
                      n = layer_dims[i], n_prev = layer_dims[i - 1],
                      activation = activations[i]))

    def model_forward(self):
        for i in range(1, self.n_layers()):
            self.layers[i].propagate_forward_from(self.layers[i - 1])        

    def model_backward(self, y):
        AL = self.layers[-1].A
        # derivative of cost with respect to final activation function
        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
        self.layers[-1].dA = dAL
        for i in reversed(range(1, self.n_layers())):
            self.layers[i].propagate_backward_to(self.layers[i - 1])

    def update_parameters(self):
        for layer in self.layers[1:]: ## skip input layer
            layer.update_parameters(self.learning_rate)
            
    def train(self, X, y, iterations = 100, debug = False):
        """ 
        Train the network.
        If there are n features and m training examples, then:
        X is a matrix n rows and m columns
        y is an array of length m
        returns an array of what the cost function's value was at each iteration
        """
        costs = []
        self.layers[0].A = X
        for i in range(iterations):
            self.model_forward()
            cost = self.compute_cost(y)
            costs.append(cost)
            if debug: print(cost)
            self.model_backward(y)
            self.update_parameters()
            if cost < 0.01:
                if debug: print("cost converged at iteration", i)
                break
        self.is_trained = True
        return costs

    def predict(self, X):
        assert(self.is_trained)
        train_input_layer = self.layers[0].A
        self.layers[0].A = X
        self.model_forward()
        yhat = self.layers[-1].A
        self.layers[0].A = train_input_layer
        return np.squeeze(yhat)
    
    def n_layers(self): 
        return len(self.layers)
    
    def compute_cost(self, y):
        m = len(y)
        AL = self.layers[-1].A
        cost =  - (1 / m) * np.sum(y * np.log(AL) + (1 - y) * np.log(1 - AL))
        return np.squeeze(cost)
    
    def __assert_ok_topology(self, l_n, l_n_minus_1):
        l_n.shape[1] == l_n_minus_1.shape[0]
