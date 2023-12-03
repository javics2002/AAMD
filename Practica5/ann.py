import numpy as np
from utils import sig

def feedForward(theta_list, a_in):
    """
    Perform feedforward propagation in a neural network.

    Parameters
    ----------
    theta_list : list of array_like
        List of weight matrices for each layer in the neural network.
        Each matrix has shape (number of units in current layer x number of units in previous layer + 1).
    a_in : array_like
        Input data with shape (number of examples x number of features).

    Returns
    -------
    activations : list of array_like
        List containing activations for each layer in the neural network.
    """
    
    m = a_in.shape[0]
    a_in = np.hstack([np.ones((m, 1)), a_in])
    activations = [a_in]
    
    for i in range(len(theta_list)):
        z = np.dot(activations[-1], theta_list[i].T)
        a = sig(z)
        
        if (i < len(theta_list) - 1):
            a = np.hstack([np.ones((m, 1)), a])
            
        activations.append(a)
        
    return activations


def predict(theta_list, a_in):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta_list : list of array_like
        List of weight matrices for each layer in the neural network.
        Each matrix has shape (number of units in current layer x number of units in previous layer + 1).
    a_in : array_like
        Input data with shape (number of examples x number of features).

    Returns
    -------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    
    activations = feedForward(theta_list, a_in)
    
    p = np.argmax(activations[-1], axis=1)

    return p

def predict_without_feedforward(out_a):
    """
    Predict the label given the activations in the output layer.

    Parameters
    ----------
    out_a : array_like
        Activations in the output layer.

    Returns
    -------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    
    p = np.argmax(out_a, axis=1)

    return p

def cost(theta_list, X, y, lambda_):
    """
    Compute the cost for a neural network.

    Parameters
    ----------
    theta_list : list of array_like
        List of weight matrices for each layer in the neural network.
        Each matrix has shape (number of units in current layer x number of units in previous layer + 1).
    X : array_like
        Input data with shape (number of examples x number of features).
    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of classes).
    lambda_ : float
        Regularization parameter.

    Returns
    -------
    J : float
        The computed value for the cost function.
    """
    
    m = len(y)
    activations = feedForward(theta_list, X)
    predictions = activations[-1]
    
    J = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    reg_term = 0
    for theta in theta_list:
        reg_term += np.sum(theta[:, 1:]**2)
    reg_term *= (lambda_ / (2 * m))
    J += reg_term
    
    return J

def backprop(theta_list, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta_list : list of array_like
        List of weight matrices for each layer in the neural network.
        Each matrix has shape (number of units in current layer x number of units in previous layer + 1).

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

    activations, zs = forward_propagation(X, theta_list)
    a_last = activations[-1]

    J = cost(theta_list, X, y, lambda_)

    deltas = [a_last - y]
    for i in range(len(theta_list) - 1, 0, -1):
        delta = np.dot(deltas[0], theta_list[i][:, 1:])*sigmoid_gradient(zs[i - 1])
        deltas.insert(0, delta)

    grads = [np.dot(d.T, a) / m for d, a in zip(deltas, activations[:-1])]

    for i in range(len(grads)):
        grads[i][:, 1:] += (lambda_ / m) * theta_list[i][:, 1:]

    grad1, grad2 = grads[0], grads[1]

    return J, grad1, grad2

