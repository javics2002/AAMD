import numpy as np
import copy
import math
import utils as data


#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # Distancia cuadrada entre el punto (x, y) y el punto que pase por y = w * x + b
    n = len(y)
    total_cost = 0

    predictions = w * x + b
    squared_errors = (predictions - y ) ** 2
    total_cost = (1 / (2 * n)) * np.sum(squared_errors)
    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    n = len(y)

    predictions = np.dot(x, w) + b
    error = predictions - y
    
    dj_dw = (1 / n) * np.dot(error, x)
    dj_db = (1 / n) * np.sum(error)
    
    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    
    w, b = w_in, b_in
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        J_history[i] = cost_function(x, y, w, b)
        grad_w, grad_b = gradient_function(x, y, w, b)
        
        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, J_history
