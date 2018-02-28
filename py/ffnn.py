
import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)    
    assert (dZ.shape == Z.shape)    
    return dZ

def initialize_parameters(layer_dims, scaling = 0.01):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = scaling * np.random.randn(layer_dims[l],
                                                             layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (W, A, b)
    return Z, cache

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    dZ = ACTIVATION_BACKWARD_MAP[activation](dA, activation_cache)
    return linear_backward(dZ, linear_cache)
    
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
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        W, b = parameters["W" + str(l)], parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activations[l])
        caches.append(cache)
    W, b = parameters["W" + str(L)], parameters["b" + str(L)]
        
    AL, cache = linear_activation_forward(A, W, b, activations[L])
    caches.append(cache)    
    return AL, caches

def L_model_backward(AL, Y, caches, activations):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    # derivative of cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    
    current_cache = caches[L - 1]
    dA, dW, db = linear_activation_backward(dAL, current_cache, 
                                            activations[L + 1])
    grads["dA" + str(L - 1)] = dA
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp =\
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activations[l + 1])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp    
    return grads
        
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost =  - (1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return np.squeeze(cost)

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters

ACTIVATION_BACKWARD_MAP = {relu: relu_backward, sigmoid: sigmoid_backward} 
    