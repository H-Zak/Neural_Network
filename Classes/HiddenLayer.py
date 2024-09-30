import numpy as np
from typing import List, Callable

class HiddenLayer():
	def __init__(self, input_shape, number_of_neurons, activation : Callable):
		self.weigths = np.random.randn(input_shape, number_of_neurons)
		self.activation = activation 

		print(self.weigths.shape)
	# def call(self, inputs, training=None):
	# 	x = np.matmul(inputs)
	# 	if self.bias is not None:
	# 		x = np.add(x, self.bias)
	# 	if self.activation is not None:
	# 		x = self.activation(x)
	# 		return x