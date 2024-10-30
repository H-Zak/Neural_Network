import numpy as np
from typing import List, Callable
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_shape, number_of_neurons, layer_type : str):
        self.weights = np.random.randn(number_of_neurons, input_shape)
        self.bias = np.zeros((number_of_neurons, 1))

    def call(self, inputs):
        Z = np.dot(self.weights, inputs) + self.bias
        A = self.activation_ft(Z)
        return A, Z
    
    @abstractmethod
    def activation_ft(self, x):
        pass

    @abstractmethod
    def derivative_activation_ft(self, output):
        pass

class HiddenLayer(Layer):
    def __init__(self, input_shape, number_of_neurons, layer_type : str):
        super().__init__(input_shape, number_of_neurons, layer_type)
        
    # def call(self, inputs):
    #     Z = np.dot(self.weights, inputs) + self.bias
    #     # print(f"------------ Zs  hidden layer ------------------------")
    #     # print(Z)
    #     A = self.activation_ft(Z)
    #     # print(f"----------- Activations  Hidden Layer ----------------")
    #     # print(A)
    #     return A, Z

    # reLU function
    def activation_ft(self, x):
        return np.maximum(0, x)

    # reLU derivative
    def derivative_activation_ft(self, x):
        # The directive of ReLU function is 1 for x > 0 and 0 for x <= 0
        return np.where(x > 0, 1, 0)

class OutputLayer(Layer):
    def __init__(self, input_shape, number_of_neurons, layer_type: str):
        super().__init__(input_shape, number_of_neurons, layer_type)

    # Softmax activation function
    def activation_ft(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerically stable softmax
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # Normalizes across classes

    # Derivative of softmax combined with cross-entropy loss
    # This simplifies the gradient calculation: softmax(output) - y_true
    def derivative_activation_ft(self, output, y_true):
        # y_true is a one-hot encoded vector
        return output - y_true
    

    # def call(self, inputs):
    #     # print(inputs)
    #     # print(self.weights)
    #     Z = np.dot(self.weights, inputs) + self.bias
    #     # print(f"------------ Zs  Output layer ------------------------")
    #     # print(Z)
    #     # print(Z.shape)
    #     A = self.activation_ft(Z)
    #     # print(f"----------- Activations  Output layer ----------------")
    #     # print(A)
    #     # print(A.shape)
    #     return A, Z