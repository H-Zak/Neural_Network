import numpy as np
from typing import List, Callable

class Layer():
    def __init__(self, shape, activation: Callable):
        self.shape = shape
        self.activation = activation
        self.bias = np.zeros(shape[1])

class HiddenLayer(Layer):
    def __init__(self, input_shape, number_of_neurons, activation: Callable):
        super().__init__(shape=(input_shape, number_of_neurons), activation=activation)
        self.number_of_neurons = number_of_neurons # It's necessary ? 
        self.weigths = np.random.randn(input_shape, number_of_neurons)

        print(f"I am a HiddenLayer and I have {self.number_of_neurons} neurones")
        print("Weigths:")
        print(self.weigths.shape)
	
    def call(self, inputs):
        x = np.dot(inputs, self.weights)
        if self.bias is not None:
            x = np.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)    
        return x