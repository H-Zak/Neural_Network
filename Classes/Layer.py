import numpy as np
from typing import List, Callable
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_shape, number_of_neurons, layer_type : str):

        if layer_type == 'hidden':
            self.weights = np.array([[ 0.14076115, -2.41981209, 0.55102647],
                                    [-0.36529239, 1.31068408, -0.68787999]])
        elif layer_type == 'output':
            self.weights =  np.array([[-0.50108095],
                                     [-1.82787485],
                                     [-0.19651201]])
        self.bias = np.zeros(number_of_neurons)
        # self.weights = np.random.randn(input_shape, number_of_neurons)
    # def call(self, inputs):
    #     Z = np.dot(inputs, self.weights) + self.bias
    #     A = self.activation_ft(Z)
    #     return A, Z
    
    @abstractmethod
    def activation_ft(self, x):
        pass

    @abstractmethod
    def derivative_activation_ft(self, output):
        pass

class HiddenLayer(Layer):
    def __init__(self, input_shape, number_of_neurons, layer_type : str):
        super().__init__(input_shape, number_of_neurons, layer_type)
        
    def call(self, inputs):
        Z = np.dot(inputs, self.weights) + self.bias
        print(f"------------ Zs  hidden layer ------------------------")
        print(Z)
        A = self.activation_ft(Z)
        print(f"----------- Activations  Hidden Layer ----------------")
        print(A)
        return A, Z

    # reLU function
    def activation_ft(self, x):
        return np.maximum(0, x)

    # reLU derivative
    def derivative_activation_ft(self, x):
        print("reLU derivative")
        # The directive of ReLU function is 1 for x > 0 and 0 for x <= 0
        return np.where(x > 0, 1, 0)

class OutputLayer(Layer):
    def __init__(self, input_shape, number_of_neurons, layer_type : str):
        super().__init__(input_shape, number_of_neurons, layer_type)

    def call(self, inputs):
        print(inputs)
        print(self.weights)
        Z = np.dot(inputs, self.weights) + self.bias
        print(f"------------ Zs  Output layer ------------------------")
        print(Z)
        A = self.activation_ft(Z)
        print(f"----------- Activations  Output layer ----------------")
        print(A)
        return A, Z

    # Sigmoid
    def activation_ft(self, x):
        return 1 / (1 + np.exp(-x))

    # def derivative_activation_ft(self, output):
    #     return output * (1 - output)

    def derivative_activation_ft(self, output):
        # print("Sigmoid derivative")
        # The derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
        sig = self.activation_ft(output)
        return sig * (1 - sig)