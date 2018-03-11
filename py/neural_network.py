#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:42:03 2018

@author: mson
"""

import activations as actv
import initializations
import numpy as np
import loss_functions


class InputLayer:
    
    def __init__(self, A):
        self.A = A
        self.name = "input"

class Layer:

    def __init__(self, name, n, n_prev, activation, use_adam = False,
                 initialization = initializations.he):
        """ n: integer; the dimension of this layer
            n_prev: integer; the dimension of the previous layer            
            activation: function; the activation function for this node
            name: the name of this node 
            initialization: function; the initialization strategy to use
        """
        self.activation = activation
        self.W = initialization(n, n_prev)
        self.b = np.zeros((n, 1))
        self.update_parameters = self.__update_adam if use_adam else self.__update_gradient_descent
        ## exponentially-weighted averages for Adam gradient descent
        self.vdW = np.zeros(self.W.shape)
        self.vdb = np.zeros(self.b.shape)
        self.sdW = np.zeros(self.W.shape)
        self.sdb = np.zeros(self.b.shape)
        
        self.name = name        
        self.A = None
        self.Z = None
    
    def shape(self): return self.W.shape
    def n_features(self): return self.shape[0]
    
    def propagate_forward_from(self, layer):
        """
        Performs forward propagation through this layer. 
        If this is layer n, then the layer argument is layer n - 1.
        """
        self.A_prev = layer.A.copy()
        self.Z = np.dot(self.W, layer.A) + self.b
        self.A = self.activation(self.Z)
        
    def propagate_backward(self):
        """
        Performs back propagation through this layer. 
        """
        m = self.A_prev.shape[1]
        dZ = actv.derivative(self.activation)(self.dA, self.Z)
        self.dW = (1 / m) * np.dot(dZ, self.A_prev.T)
        self.db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
        return np.dot(self.W.T, dZ) ## this is dA_prev
    
    def __update_gradient_descent(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
    
    def __update_adam(self, learning_rate, t,
               beta1, beta2, eps = 1e-8):
        """ 
        update parameters W and b using Adam gradient descent.
        """
        assert(0 <= learning_rate <= 1
               and 0 <= beta1 <= 1
               and 0 <= beta2 <= 1
               and t > 0)
        self.vdW = (beta1 * self.vdW + (1 - beta1) * self.dW) / (1 - beta1**t)
        self.vdb = (beta1 * self.vdb + (1 - beta1) * self.db) / (1 - beta1**t)
        self.sdW = (beta2 * self.sdW + (1 - beta2) * (self.dW**2)) / (1 - beta2**t)
        self.sdb = (beta2 * self.sdb + (1 - beta2) * (self.db**2)) / (1 - beta2**t)
        self.W -= learning_rate * (self.vdW / (np.sqrt(self.sdW) + eps))
        self.b -= learning_rate * (self.vdb / (np.sqrt(self.sdb) + eps))

class Net:
    """ A Net is made of layers
    """
    def __init__(self, layer_dims, activations,
                 loss = loss_functions.LogLoss(),
                 use_adam = False):
        """
        layer_dims: an array of layer dimensions. 
                    including the input layer.
        activations: an array of activation 
                     functions (each from the activations module); 
                     one function per layer
        loss: the cost function. 
        """
        assert(len(layer_dims) == len(activations))

        self.use_adam = use_adam
        self.is_trained = False
        self.J = loss.cost
        self.J_prime = loss.cost_gradient
        
        self.hidden_layers = []
        for i in range(1, len(layer_dims)):
            self.hidden_layers.append(
                Layer(name = i,
                      n = layer_dims[i], n_prev = layer_dims[i - 1],
                      activation = activations[i],
                      use_adam = use_adam))

    def __model_forward(self, input_layer):
        """ Does one full forward pass through the network. """
        
        self.hidden_layers[0].propagate_forward_from(input_layer)
        for i in range(1, self.n_layers()):
            self.hidden_layers[i].propagate_forward_from(self.hidden_layers[i - 1])

    def shape(self):
        return [l.W.shape for l in self.hidden_layers]
            
    def __model_backward(self, y):
        """ Does one full backward pass through the network. """
        AL = self.hidden_layers[-1].A
        # derivative of cost with respect to final activation function
        dA_prev = self.J_prime(AL, y)
        for layer in reversed(self.hidden_layers):
            layer.dA = dA_prev
            dA_prev = layer.propagate_backward()
            
    def __update_parameters(self, learning_rate):
        """ Updates parameters on each layer at epoch t. """
        for layer in self.hidden_layers:
            layer.update_parameters(learning_rate)
            
    def __adam(self, learning_rate, t, beta1, beta2):
        """ Updates parameters on each layer at epoch t. """
        for layer in self.hidden_layers:
            layer.update_parameters(learning_rate, t, beta1, beta2)
            
    def train(self, X, y, iterations = 100, learning_rate = 0.01,
              beta1 = 0.9, beta2 = 0.99,
              debug = False):
        """ 
        Train the network.
        -- Arguments:
        If there are n features and m training examples, then:
        X: a matrix n rows and m columns
        y: an array of length m
        returns an array of what the cost function's value was at each iteration
        """
        costs = []
        input_layer = InputLayer(X)
        AL = self.hidden_layers[-1].A
        for i in range(1, iterations + 1):
            self.__model_forward(input_layer)
            yhat = self.hidden_layers[-1].A
            cost = self.J(yhat, y)
            costs.append(cost)
            if debug: print(cost)
            self.__model_backward(y)
            if self.use_adam:
                self.__adam(learning_rate, t = i, beta1 = beta1, beta2 = beta2)
            else:
                self.__update_parameters(learning_rate)
            if cost < 0.01:
                if debug: print("cost converged at iteration", i)
                break
        self.is_trained = True
        return costs

    def predict(self, X):
        assert(self.is_trained)
        self.__model_forward(InputLayer(X))
        yhat = self.hidden_layers[-1].A
        return np.squeeze(yhat)
    
    def n_layers(self): 
        return len(self.hidden_layers)
    
    def __gradient_check(self, eps = 1e-7):
        """ not finished """
        W_vec  = self.__stack_things(lambda lyr: self.__matrix_to_vector(lyr.W))
        dW_vec = self.__stack_things(lambda lyr: self.__matrix_to_vector(lyr.dW))
        b_vec  = self.__stack_things(lambda lyr: lyr.b.reshape(lyr.b.shape[0]))
        db_vec = self.__stack_things(lambda lyr: lyr.db.reshape(lyr.db.shape[0]))

    def __approximate_derivative(self, vec, i, eps):
        """ not finished """
        vec[i] += eps

    @staticmethod
    def __matrix_to_vector(mat):
        """ reshape m-by-n matrix into an m*n-length array"""
        vec_len = mat.shape[0] * mat.shape[1]
        return mat.reshape(vec_len,)
    
    def __stack_things(self, action_fn):
        """ apply action_fn to each layer in hidden_layers 
            and concatenate the results into a single vector
        """
        return np.concatenate([action_fn(l) for l in self.hidden_layers])