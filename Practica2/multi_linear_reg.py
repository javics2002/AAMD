import numpy as np
import copy
import math

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Dividir la línea en valores y convertirlos a números
            values = line.strip().split(',')
            size = float(values[0])
            bedrooms = float(values[1])
            floors = float(values[2])
            age = float(values[3])
            price = float(values[4])
            # Crear una tupla con los valores y agregarla a la lista de datos
            data.append([size, bedrooms, floors, age, price])
    return data

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)

    X_norm = (X - mu) / sigma
    
    return (X_norm, mu, sigma)


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    n = len(y)
    cost = 0

    predictions = np.dot(X, w) + b
    squared_errors = (predictions - y) ** 2
    cost = (1 / (2 * n)) * np.sum(squared_errors)
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """
    
    n = len(y)

    predictions = np.dot(X, w) + b
    
    error = predictions - y
    
    dj_dw = (1 / n) * np.dot(X.T, error)
    dj_db = (1 / n) * np.sum(error)

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model
          after running gradient descent
      b : (scalar)                Updated value of parameter of the model 
          after running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        J_history[i] = cost_function(X, y, w, b)
        grad_w, grad_b = gradient_function(X, y, w, b)
        
        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, J_history
