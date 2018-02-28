
import numpy as np
def initialize_parameters(layer_dims, scaling = 0.01):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters[l] = {'W': scaling * np.random.randn(layer_dims[l],
                                                        layer_dims[l - 1]),\
                         'b': np.zeros((layer_dims[l], 1))}
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (W, A, b)
    return Z, cache

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments: A_prev, W: matrices such that dot(W, A_prev) is valid
    b: a vector
    activation: a function to use as the activation function
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, activations):
    caches = []
    A = X
    L = len(parameters)
    
    for l in range(1, L):
        A_prev = A 
        W, b = parameters[l]['W'], parameters[l]['b']
        A, cache = linear_activation_forward(A_prev, W, b, activations[l])
        caches.append(cache)
    W, b = parameters[L]['W'], parameters[L]['b']
        
    AL, cache = linear_activation_forward(A, W, b, activations[L])
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost =  - (1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return np.squeeze(cost)
    
    
    