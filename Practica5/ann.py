import numpy as np

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


	return (J, grad1, grad2)

