#!/usr/bin/env python3
"""Class Neuron that defines a single neuron"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """Neuron class that performs binary classification"""

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        # initializes the weights __W as a
        # numpy array of shape (1, nx) with random normal values.
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        # Calculate the linear combination of inputs and weights, plus bias
        Z = np.matmul(self.__W, X) + self.__b

        # Apply the sigmoid activation function to the linear combination
        self.__A = self.sigmoid(Z)
        return self.__A

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
        """Evaluates the neurons predictions"""
        # Calculate the activated output of the neuron given the input data
        A = self.forward_prop(X)

        # Apply a threshold of 0.5 to the
        # activated output to get binary predictions
        prediction = np.where(A >= 0.5, 1, 0)

        # Calculate the cost of the model's
        # predictions using the actual labels
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        #  Get the number of examples in the training set
        m = Y.shape[1]

        # Compute the derivative of the cost function
        # with respect to the activated output
        dz = A - Y

        # Compute the derivative of the cost function
        dw = np.matmul(X, dz.T) / m

        # Compute the derivative of the cost function
        # with respect to the bias
        db = np.sum(dz) / m
        # Update the weights using the gradient descent algorithm
        self.__W -= alpha * dw.T

        # Update the bias using the gradient descent algorithm
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron with the given input data and correct labels.
        """
        # Validate inputs
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        costs = []
        steps = []
        # Training loop
        for i in range(iterations + 1):
            # Forward propagation
            A = self.forward_prop(X)
            # Calculate the cost
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                # Backward propagation
                self.gradient_descent(X, Y, A, alpha)
        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iterations")
            plt.ylabel("cost")
            plt.title("Training Cost")

        # Evaluate predictions and cost
        predictions, cost = self.evaluate(X, Y)

        return predictions, cost
