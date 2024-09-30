import numpy as np
from typing import List, Callable
from Classes.HiddenLayer import HiddenLayer

class NeuralNetwork():
	def __init__(self, inputs : np.ndarray ,hidden_layers : List[int], activation : Callable):
		self.inputs : np.ndarray = inputs
		self.hidden_layers : List[HiddenLayer] = []
		self.activation = activation

		for i in range(len(hidden_layers)):
			print(f"Layer {i}:")
			self.hidden_layers.append(HiddenLayer(self.inputs.shape[1], hidden_layers[i], self.activation))
	
	# def call(self):
		