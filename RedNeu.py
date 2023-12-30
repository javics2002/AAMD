import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize

# Función de activación sigmoidal
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradiente de la función sigmoidal
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Función de costo con regularización
def cost(theta_list, X, Y, lambda_):
    m = len(Y)
    predictions = predict(theta_list, X)
    J = (-1 / m) * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
    reg_term = 0
    for theta in theta_list:
        reg_term += np.sum(theta[:, 1:]**2)
    reg_term *= (lambda_ / (2 * m))
    J += reg_term
    return J

# Función de predicción
def predict(theta_list, X):
    m = X.shape[0]
    activations = [X]
    for i in range(len(theta_list) - 1):
        z = np.dot(activations[-1], theta_list[i].T)
        a = sigmoid(z)
        a = np.hstack([np.ones((m, 1)), a])
        activations.append(a)
    # Última capa sin añadir sesgo
    z_last = np.dot(activations[-1], theta_list[-1].T)
    a_last = sigmoid(z_last)
    activations.append(a_last)
    return activations[-1]

# Inicializar parámetros de la red neuronal con capas y neuronas personalizables
def initialize_parameters(layer_sizes):
    theta_list = []
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        epsilon = 0.12
        theta = np.random.rand(output_size, input_size + 1) * 2 * epsilon - epsilon
        theta_list.append(theta)
    return theta_list

# Función de forward propagation
def forward_propagation(X, theta_list):
    activations = [X]
    m = X.shape[0]
    zs = []
    for i in range(len(theta_list) - 1):
        z = np.dot(activations[-1], theta_list[i].T)
        a = sigmoid(z)
        a = np.hstack([np.ones((m, 1)), a])
        activations.append(a)
        zs.append(z)
    # Última capa sin añadir sesgo
    z_last = np.dot(activations[-1], theta_list[-1].T)
    a_last = sigmoid(z_last)
    activations.append(a_last)
    zs.append(z_last)
    return activations, zs

def update_parameters(theta_list, grads, learning_rate):
    for i in range(len(theta_list)):
        theta_list[i] -= learning_rate * grads[i]



# Función de backpropagation
def backprop(theta_list, X, y, lambda_):
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
        delta = np.dot(deltas[0], theta_list[i][:, 1:]) * sigmoid_gradient(zs[i - 1])
        deltas.insert(0, delta)

    # Gradients
    grads = [np.dot(d.T, a) / m for d, a in zip(deltas, activations[:-1])]

    # Regularization term gradients (excluding bias terms)
    for i in range(len(grads)):
        grads[i][:, 1:] += (lambda_ / m) * theta_list[i][:, 1:]


    return J, grads