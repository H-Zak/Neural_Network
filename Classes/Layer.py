import numpy as np
from typing import List, Callable
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def __init__(self, count_neurons):
        self.bias = np.zeros(count_neurons)
    
    def update_weights(self):
        pass

class HiddenLayer(Layer):
    def __init__(self, input_shape, number_of_neurons):
        super().__init__(number_of_neurons)
        
        # Randow weights
        self.weights = np.random.randn(input_shape, number_of_neurons)

    def call(self, inputs):
        Z = np.dot(inputs, self.weights) + self.bias
        A = self.activation_ft(Z)
        return A, Z

    # reLU function
    def activation_ft(self, x):
        return np.maximum(0, x)

    # reLU derivative
    def derivative_activation_ft(self, x):
        # The directive of ReLU function is 1 for x > 0 and 0 for x <= 0
        return np.where(x > 0, 1, 0)

class OutputLayer(Layer):
    def __init__(self, input_shape, number_of_neurons):
        super().__init__(number_of_neurons)
        
        # Randow weights
        self.weights = np.random.randn(input_shape, number_of_neurons)

    def call(self, inputs):
        Z = np.dot(inputs, self.weights) + self.bias
        print(Z)
        print("-----------------------------------------")
        A = self.activation_ft(Z)
        return A, Z

    # Softmax function
    def activation_ft(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Softmax derivative
    def derivative_activation_ft(output):
        s = output.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    # def sigmoid_derivative(self, x):
    #     # The derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
    #     sig = self.sigmoid(x)
    #     return sig * (1 - sig)