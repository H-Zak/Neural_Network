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
            self.layers.append(HiddenLayer(prev_shape, layer_size, f'hidden {i+1}'))
            prev_shape = layer_size 

            # print("Weights shape:")
            # print(self.layers[i].weights.shape)
            # print(self.layers[i].weights)
            # print("Bias shape:")
            # print(self.layers[i].bias.shape)

        # print(f"Output layer of {output_shape} neurons")
        self.layers.append(OutputLayer(prev_shape, output_shape, f'output'))
        # print("Weights shape (output layer):")
        # print(self.layers[-1].weights.shape)
        # print("Bias shape (output layer):")
        # print(self.layers[-1].bias.shape)

    def feedforward(self, inputs):
        A = inputs
        for i in range(len(self.layers)):
            A = self.layers[i].forward(A)
        return A

    def backpropagation(self, inputs, outputs, Y, learning_rate):
        m = inputs.shape[1]
        # Derivative of loss function
        dA = outputs - Y
        for layer in reversed(self.layers):
            dZ = layer.backward(dA)
            dW, db = layer.get_gradients()
            layer.update_parameters(dW, db, learning_rate)
            dA = dZ

    def test(self, X, y, epochs):
        # Assuming the implementation of the forward and backward passes is already in place
        for epoch in range(1, epochs + 1):
            # Forward pass
            A = X
            for i in range(len(self.layers)):
                A = self.layers[i].forward(A)

            # Compute loss (dummy implementation, replace with your actual loss function)
            loss = self.compute_loss(y, A)
            print(f"epoch {epoch}/{epochs} - loss: {loss:.4f} - val_loss: {0}")  # Placeholder for val_loss

            # Backward pass
            dA = A - y  # Assuming A is the output and y is the target
            for i in reversed(range(len(self.layers))):
                dZ = self.layers[i].backward(dA)
                print(f"dA shape: {dA.shape}")
                print(f"dZ shape (layer {i}): {dZ.shape}")
                dW, db = self.layers[i].get_gradients()
                print(f"dW[{i}] shape: {dW.shape}")
                print(f"db[{i}] shape: {db.shape}")
                dA = dZ

                # Update weights and biases
                self.layers[i].update_parameters(dW, db)
                print(f"weights[{i}] shape after update: {self.layers[i].weights.shape}")
                print(f"bias[{i}] shape after update: {self.layers[i].bias.shape}")

    
    # def feedforward(self, inputs):
    #     self.activations = []
    #     self.Zs = []
        
    #     A = inputs
    #     self.activations.append(A)
    #     for i, layer in enumerate(self.layers):
    #         A, Z = layer.call(A)
    #         self.Zs.append(Z)
    #         self.activations.append(A)
    #     return A

    # def backpropagation(self, inputs, outputs, Y, learning_rate):
    #     m = inputs.shape[1]
    #     # Derivative arrays
    #     dW = [None] * len(self.layers)
    #     db = [None] * len(self.layers)

    #     # Derivative of binary_cross_entropy loss function
    #     dA = outputs - Y
    #     print(f"dA shape: {dA.shape}")  # Debugging: Ver la forma de dA
    #     for idx in reversed(range(len(self.layers))):
            
    #         if idx == len(self.layers) - 1:
    #             dZ = dA
    #         else:
    #             # dL/dZ = dL/dA⋅softmax′(Z)
    #             dZ = dA * self.layers[idx].derivative_activation_ft(self.Zs[idx])
            
    #         print(f"dZ shape (layer {idx}): {dZ.shape}")  # Debugging: Ver la forma de dZ

    #         dW[idx] = (1/m) * np.dot(dZ, self.activations[idx].T)
    #         print(f"dW[{idx}] shape: {dW[idx].shape}")  # Debugging: Ver la forma de dW

    #         # Compute of the gradient with respect to the bias (db)
    #         db[idx] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    #         print(f"db[{idx}] shape: {db[idx].shape}")  # Debugging: Ver la forma de db
        
    #         # dA for the hidden layer | update
    #         if idx > 0:
    #             dA = np.dot(self.layers[idx].weights.T, dZ)
    #             print(f"dA shape (after layer {idx}): {dA.shape}")  # Debugging: Ver la forma de dA

    #         # Actualización de los pesos y biases
    #         self.layers[idx].weights -= learning_rate * dW[idx]
    #         print(f"weights[{idx}] shape after update: {self.layers[idx].weights.shape}")  # Debugging: Ver la forma de weights
    #         self.layers[idx].bias -= learning_rate * db[idx]
    #         print(f"bias[{idx}] shape after update: {self.layers[idx].bias.shape}")  # Debugging: Ver la forma de bias
    
    # def backpropagation(self, inputs, outputs, Y, learning_rate):
    #     m = inputs.shape[1]
    #     # Derivative arrays
    #     # dA = [None] * len(self.layers)
    #     dW = [None] * len(self.layers)
    #     db = [None] * len(self.layers)

    #     # Derivative of binary_cross_entropy loss function
    #     dA = outputs - Y
    #     for idx in reversed(range(0, len(self.layers), 1)):
            
    #         if idx == len(self.layers) - 1:
    #             dZ = dA
    #         else:
    #             # dL/dZ = dL/dA⋅softmax′(Z)
    #             dZ = dA * self.layers[idx].derivative_activation_ft(self.Zs[idx])
    #         # print("dZ shape")
    #         # print(dZ.shape)


    #         dW[idx] = (1/m) * np.dot(dZ, self.activations[idx].T)

    #         # Compute of the gradient with respect to the bias (db)
    #         # axis=0 or 1 ? 
    #         db[idx] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        

    #         # print(db[idx].shape)
    #         # dA for the hidden layer | update
    #         if idx > 0:
    #             dA = np.dot(self.layers[idx].weights.T, dZ)
    #             # print("dA shape")
    #             # print(dA.shape)
    #         self.layers[idx].weights = self.layers[idx].weights - learning_rate * dW[idx]
    #         self.layers[idx].bias = self.layers[idx].bias - learning_rate * db[idx]
