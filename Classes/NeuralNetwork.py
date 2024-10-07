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

            print("Weights:")
            print(self.layers[i].weights)
            print("Bias:")
            print(self.layers[i].bias)

        # print(f"Output layer of {output_shape} neurons")
        self.layers.append(OutputLayer(prev_shape, output_shape))
        print("Weights (output layer):")
        print(self.layers[-1].weights)
        print("Bias (output layer):")
        print(self.layers[-1].bias)
    
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

        print(len(self.activations))
        print(len(self.Zs))
        return A
    
    def backpropagation(self, outputs, Y):
        # print("--------- Outputs ------------")
        # print(outputs)
        # print("--------- Y ------------")
        # print(Y)
        dA = outputs - Y
        for idx in reversed(range(0, len(self.layers), 1)):
            # print(idx)
            # dL/dZ=dL/dA⋅softmax′(Z)
            dZ = dA * self.layers[idx].derivative_activation_ft(self.Zs[idx])
            # print("-------------- dZs -----------------")
            # print(dZ)
            print(f"----------- Activations layer {idx} ----------------")
            print(self.activations[idx])
            # Compute of the gradient with respect to the weights (dW)
            dW = np.dot(self.activations[idx-1].T, dZ)
            # Compute of the gradient with respect to the bias (db)
            db = np.sum(dZ, axis=0, keepdims=True)


            # if idx > 0:
            #     input_to_layer = self.layers[idx - 1].A
            # else:
            #     input_to_layer = self.inputs

