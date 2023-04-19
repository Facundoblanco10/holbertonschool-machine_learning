#!/usr/bin/env python3
"""
Class NeuralNetwork that defines a neural network
with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork():
    """Class NeuralNetwork"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        # Calculate the weighted sum of the input values
        # forr each neuron in the hidden layer
        Z1 = np.matmul(self.__W1, X) + self.__b1

        # Apply the sigmoid activation function to the weighted
        # sum to get the output of the hidden layer
        self.__A1 = self.sigmoid(Z1)

        # Calculate the weighted sum of the output values from
        # the hidden layer forr the single neuron in the output layer
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2

        # Apply the sigmoid activation function to the weighted
        # sum to get the final output of the neural network
        self.__A2 = self.sigmoid(Z2)

        return self.__A1, self.__A2

    def sigmoid(self, z):
        """
        The sigmoid function is defined as 1 / (1 + e^(-z)),
        where e is Euler's number and z is the input value. It takes
        any real-valued number and maps it to a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        # Get the number of examples in the training set
        m = Y.shape[1]

        # Compute the cost using the logistic regression cost function
        cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural networks predictions"""
        # Forward propagate X through the neural network to
        # obtainthe activation of the last layer
        _, A2 = self.forward_prop(X)

        # Threshold the activation to obtain binary
        # predictions (0 or 1) using a threshold of 0.5
        predictions = np.where(A2 >= 0.5, 1, 0)

        # Calculate the cost of the predictions using
        # the cost function defined in the neural network
        cost = self.cost(Y, A2)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Calculates one pass of gradient descent
            on the neural network
        """
        # Get the number of training examples
        m = Y.shape[1]

        # Compute the error in the final layer
        dz2 = A2 - Y

        # Compute the gradient of the weights in the final layer
        dw2 = np.matmul(dz2, A1.T) / m

        # Compute the gradient of the bias in the final layer
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # Compute the error in the first hidden layer
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))

        # Compute the gradient of the weights in the first hidden layer
        dw1 = np.matmul(dz1, X.T) / m

        # Compute the gradient of the bias in the first hidden layer
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # Update the weights and biases in the neural network
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1