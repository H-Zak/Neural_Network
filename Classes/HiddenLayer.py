import numpy as np
from typing import List, Callable

class Layer():
    def __init__(self, shape, activation_ft: Callable):
        # self.shape = shape
        self.activation_ft = activation_ft
        self.bias = np.zeros(shape)

class HiddenLayer(Layer):
    def __init__(self, input_shape, number_of_neurons, activation_ft: Callable):
        super().__init__(number_of_neurons, activation_ft=activation_ft)
        
        self.number_of_neurons = number_of_neurons
        self.weights = np.random.randn(input_shape, self.number_of_neurons)

    def call(self, inputs):
        self.print_weights_info()
        self.print_bias_info()
        Z = np.dot(inputs, self.weights) + self.bias
        if self.activation_ft is None:
            raise("Error the activation function is not defined!")
        A = self.activation_ft(Z)
        return A, Z

    def print_weights_info(self):
        print("Weights shape:")
        print(self.weights.shape)
        print("Weights values")
        print(self.weights)

    def print_bias_info(self):
        print("Bias shape:")
        print(self.bias.shape)
        print("Bias values")
        print(self.bias)