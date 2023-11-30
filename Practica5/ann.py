import numpy as np
from utils import sig

def feedForward(theta1, theta2, a1):
    m = a1.shape[0]
    a1s = np.hstack([np.ones((m, 1)), a1])
    
    a1 = np.column_stack((np.ones((a1.shape[0], 1)), a1))
    
    a2 = sig(np.dot(a1s, theta1.T))
    
    a2 = np.hstack([np.ones((m, 1)), a1])
    
    a3 = sig(np.dot(a1, theta2.T))
    
    return a1, a2, a3


def predict(theta1, theta2, a1):
    """
    Predict the label of an input given a trained neural network.
    
    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    theta2: array_like
        Weights for the second layer in the neural network.
        It has shape (output layer size x 2nd hidden layer size)
    a1 : array_like
        The image inputs having shape (number of examples x image dimensions).
    
    Return
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    
    a1, a2, a3 = feedForward(theta1, theta2, a1)
    
    p = np.argmax(a3, axis=1)

    return p

def predict_without_feedforward(a3):
    p = np.argmax(a3, axis=1)

    return p

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """

    m = len(y)
    
    predictions = predict(theta1, theta2, X)
    
    J = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    reg_term = 0
    
    reg_term += np.sum(theta1 ** 2)
    reg_term += np.sum(theta2 ** 2)
    
    #reg_term += np.sum(theta1[:, 1:] ** 2)
    #reg_term += np.sum(theta2[:, 1:] ** 2)
        
    reg_term *= (lambda_ / (2 * m))
    
    J += reg_term
    
    return J

def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """

    m = X.shape[0]
    n_labels = theta_list[-1].shape[0]

    # Forward propagation
    activations, zs = forward_propagation(X, theta_list)
    a_last = activations[-1]

    # Cost calculation
    J = cost(theta_list, X, y, lambda_)


    # Backpropagation
    deltas = [a_last - y]
    for i in range(len(theta_list) - 1, 0, -1):
        delta = np.dot(deltas[0], theta_list[i][:, 1:])*sigmoid_gradient(zs[i - 1])
        deltas.insert(0, delta)

    # Gradients
    grads = [np.dot(d.T, a) / m for d, a in zip(deltas, activations[:-1])]

    # Regularization term gradients (excluding bias terms)
    for i in range(len(grads)):
        grads[i][:, 1:] += (lambda_ / m) * theta_list[i][:, 1:]

    # Rename gradients
    grad1, grad2 = grads[0], grads[1]

    return J, grad1, grad2

