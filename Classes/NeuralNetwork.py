import numpy as np
from typing import List
from Classes.Layer import Layer,HiddenLayer, OutputLayer

class NeuralNetwork():
    def __init__(self, input_shape: int, hidden_layers: List[int], output_shape: int):
        self.input_shape: int = input_shape
        self.layers: List[Layer] = []

        self.activations = []
        self.Zs = []

        prev_shape = self.input_shape

        for i, layer_size in enumerate(hidden_layers):
            # print(f"Hidden layer of {layer_size} neurons")
            self.layers.append(HiddenLayer(prev_shape, layer_size))
            prev_shape = layer_size 

        #     print("Weights:")
        #     print(self.layers[i].weights)
        #     print("Bias:")
        #     print(self.layers[i].bias)

        # print(f"Output layer of {output_shape} neurons")
        self.layers.append(OutputLayer(prev_shape, output_shape))
        # print("Weights (output layer):")
        # print(self.layers[-1].weights)
        # print("Bias (output layer):")
        # print(self.layers[-1].bias)
    
    def feedforward(self, inputs):
        self.activations = []
        self.Zs = []
        
        A = inputs
        self.activations.append(A)
        for i, layer in enumerate(self.layers):
            # print(f"Layer {i + 1}")
            A, Z = layer.call(A)
            # print(f"Z (Layer {i + 1}): {Z}")
            # print(f"A (Layer {i + 1}): {A}")
            self.Zs.append(Z)
            self.activations.append(A)
        return A
    
    def backpropagation(self, outputs, y):
        # Creating array of zeros as output shape
        y_true = np.zeros_like(outputs)
        # Updating y_true to contain a 1 in the position corresponding to the correct class for each entry in outputs
        y_true[range(len(outputs)), y] = 1
        # Derivative of softmax
        dA = outputs - y_true
        for idx in reversed(range(0, len(self.layers), 1)):
            # dL/dZ=dL/dA⋅softmax′(Z)
            dZ = dA - self.layers[idx].derivative_activation_ft(self.Zs[idx])

