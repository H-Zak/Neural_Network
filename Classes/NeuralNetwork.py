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
            self.layers.append(HiddenLayer(prev_shape, layer_size, 'hidden'))
            prev_shape = layer_size 

            # print("Weights shape:")
            # print(self.layers[i].weights.shape)
            # print(self.layers[i].weights)
            # print("Bias shape:")
            # print(self.layers[i].bias.shape)

        # print(f"Output layer of {output_shape} neurons")
        self.layers.append(OutputLayer(prev_shape, output_shape, 'output'))
        # print("Weights shape (output layer):")
        # print(self.layers[-1].weights.shape)
        # print("Bias shape (output layer):")
        # print(self.layers[-1].bias.shape)
    
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
    
    # def backpropagation(self, inputs, outputs, Y, learning_rate):
    #     m = inputs.shape[1]
    #     # Derivative arrays
    #     # dA = [None] * len(self.layers)
    #     dW = [None] * len(self.layers)
    #     db = [None] * len(self.layers)

    #     # Derivative of binary_cross_entropy loss function
    #     dA = outputs - Y
    #     # print(dA.shape)

    #     # print(len(self.Zs))
    #     # print(len(self.activations))
    #     for idx in reversed(range(0, len(self.layers), 1)):
    #         # print(f"Layer {idx}")

    #         # dL/dZ = dL/dA⋅softmax′(Z)
    #         dZ = dA * self.layers[idx].derivative_activation_ft(self.Zs[idx])
    #         # if idx == 0:
    #         #     print(self.Zs[idx])

    #         # Compute of the gradient with respect to the weights (dW) -> TODO -> See
    #         dW[idx] = (1/m) * np.dot(dZ, self.activations[idx].T) # activation[1]
    #         # Compute of the gradient with respect to the bias (db)
    #         # axis=0 or 1 ? 
    #         db[idx] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
    #         # print(f" Weights {self.layers[idx].weights.shape}")
    #         # print(f" Bias {self.layers[idx].bias.shape}")

    #         self.layers[idx].weights = self.layers[idx].weights - learning_rate * dW[idx]
    #         self.layers[idx].bias = self.layers[idx].bias - learning_rate * db[idx]
            
    #         # dA for the hidden layer
    #         dA = np.dot(self.layers[idx].weights.T, dZ)


    def backpropagation(self, inputs, outputs, Y, learning_rate):
        m = inputs.shape[1]  # Número de muestras en el batch
        
        # Derivative arrays
        dW = [None] * len(self.layers)
        db = [None] * len(self.layers)

        # Derivative of cross-entropy loss function for softmax (dA = softmax(outputs) - Y)
        dA = outputs - Y
        
        # Backpropagation loop (iterate over layers in reverse order)
        for idx in reversed(range(0, len(self.layers))):
            # If it's the ouput layer (softmax), dA is already
            if idx == len(self.layers) - 1:
                dZ = dA # (softmax - Y)
            else:
                dZ = dA * self.layers[idx].derivative_activation_ft(self.Zs[idx])
            
            # Compute gradient with respect to the weights (dW)
            dW[idx] = (1/m) * np.dot(dZ, self.activations[idx].T)
            # Compute gradient with respect to the bias (db)
            db[idx] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            # Update weights and biases
            self.layers[idx].weights -= learning_rate * dW[idx]
            self.layers[idx].bias -= learning_rate * db[idx]
            
            # # Propagate dA for the next layer (except for the last layer)
            # if idx > 0:
            dA = np.dot(self.layers[idx].weights.T, dZ)
