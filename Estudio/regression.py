import numpy as np

# Regresion lineal
def mse(real_y, estimated_y):
    """
    Calculate the Mean Squared Error between the real and estimated values.

    Parameters
    ----------
    real_y : array_like
        Array of real (actual) values.
    estimated_y : array_like
        Array of estimated values.

    Returns
    -------
    J : float
        The Mean Squared Error.
    """

    m = len(real_y)
    J = (1 / (2 * m)) * np.sum((real_y - estimated_y) ** 2)
    return J

def linear_regression(x, w, b):
    """
    Perform linear regression based on the input, weight, and bias.

    Parameters
    ----------
    x : array_like
        Input data.
    w : float
        Weight parameter.
    b : float
        Bias parameter.

    Returns
    -------
    y : array_like
        Estimated values.
    """

    return np.dot(x, w) + b

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

    predictions = linear_regression(x, w, b)
    return mse(y, predictions)

# Descenso de gradiente
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
    
    m = len(y)

    predictions = linear_regression(x, w, b)
    error = predictions - y
    
    dj_dw = (1 / m) * np.dot(error, x)
    dj_db = (1 / m) * np.sum(error)
    
    return dj_dw, dj_db


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
      w_history : (ndarray): Shape (num_iters,) w at each iteration,
          primarily for graphing later
      b_history : (ndarray): Shape (num_iters,) b at each iteration,
          primarily for graphing later
    """
    
    w, b = w_in, b_in
    w_history = np.zeros(num_iters)
    b_history = np.zeros(num_iters)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        w_history[i] = w
        b_history[i] = b
        J_history[i] = cost_function(x, y, w, b)
        grad_w, grad_b = gradient_function(x, y, w, b)
        
        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, J_history, w_history, b_history

def stochastic_gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs stochastic gradient descent to learn theta. Updates theta by taking 
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
      w_history : (ndarray): Shape (num_iters,) w at each iteration,
          primarily for graphing later
      b_history : (ndarray): Shape (num_iters,) b at each iteration,
          primarily for graphing later
    """
    
    w, b = w_in, b_in
    w_history = np.zeros(num_iters)
    b_history = np.zeros(num_iters)
    J_history = np.zeros(num_iters)
    m = len(y)
    
    for i in range(num_iters):
        w_history[i] = w
        b_history[i] = b
        J_history[i] = cost_function(x, y, w, b)
        rand_index = np.random.randint(m)
        x_i, y_i = x[rand_index], y[rand_index]
        grad_w, grad_b = gradient_function([x_i], [y_i], w, b)
        
        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, J_history, w_history, b_history

def mini_batch_gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, batch_size):
    """
    Performs mini-batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
      batch_size : (int) Size of the mini-batch
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
      w_history : (ndarray): Shape (num_iters,) w at each iteration,
          primarily for graphing later
      b_history : (ndarray): Shape (num_iters,) b at each iteration,
          primarily for graphing later
    """
    
    w, b = w_in, b_in
    w_history = np.zeros(num_iters)
    b_history = np.zeros(num_iters)
    J_history = np.zeros(num_iters)
    m = len(y)
    
    for i in range(num_iters):
        w_history[i] = w
        b_history[i] = b
        J_history[i] = cost_function(x, y, w, b)
        rand_indices = np.random.choice(m, batch_size, replace=False)
        x_batch, y_batch = x[rand_indices], y[rand_indices]
        grad_w, grad_b = gradient_function(x_batch, y_batch, w, b)
        
        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, J_history, w_history, b_history
