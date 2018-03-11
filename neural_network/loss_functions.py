"""
Loss functions for neural networks.
"""

import numpy as np

class LogLoss:
    @staticmethod
    def cost(yhat, y):
        """ 
        yhat: matrix with predictions
        y: array of true labels
        returns the scalar value of the log loss.
        """
        m = len(y)
        cost =  - (1 / m) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return np.squeeze(cost)

    # derivative of cost with respect to final activation function
    @staticmethod
    def cost_gradient(yhat, y):
        """ 
        returns the gradient w.r.t. yhat of predicting the column matrix yhat
        when true labels are the array y. 
        """
        return -(np.divide(y, yhat) - np.divide(1 - y, 1 - yhat))

class MSE:
    @staticmethod
    def cost(yhat, y):
        """ 
        yhat: matrix with predictions
        y: array of true values
        """
        m = len(y)
        cost = (1 / m) * np.sum((yhat - y)**2)
        return np.squeeze(cost)

    @staticmethod
    def cost_gradient(yhat, y):
        ## this is incorrect: it returns a scalar, whereas we need a gradient vector
        m = len(y)        
        return (2 / m) * np.sum((yhat - y) * y)