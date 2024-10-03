import numpy as np
from typing import List, Callable
from Classes.HiddenLayer import HiddenLayer
from modules.activation_funcs import relu

class NeuralNetwork():
	def __init__(self,
			  	input_shape : int,
				hidden_layers : List[int],
				activation : Callable):
		
		self.input_shape : int = input_shape
		self.hidden_layers : List[HiddenLayer] = []
		self.activation_ft = activation

		self.activations = []
		self.Zs = []

		prev_shape = self.input_shape
		for layer_size in hidden_layers:
			self.hidden_layers.append(HiddenLayer(prev_shape, layer_size, activation))
			prev_shape = layer_size

	def feedforward(self, inputs):
		A = inputs
		self.activations.append(A)
		for i, hidden_layer in enumerate(self.hidden_layers):
			print(f"Layer {i + 1}")
			A, Z = hidden_layer.call(A)
			self.Zs.append(Z)
			self.activations.append(A)
		return A

	def backpropagation(self):
		print("Backpropagation!")

		