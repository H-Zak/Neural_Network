import numpy as np

class NeuralNetwork():
	def __init__(self, layers):
		print("Start Class NeuralNetork")
		# Iniatialise un reseau de neurones.
		# :param layers : liste contenant le nombre de neuronnes dans chaque layers
		if not isinstance(layers,(tuple, list)):
			raise ValueError("layers must be a tuple or a list")
		if len(layers) < 2:
			raise ValueError("layers must had at least input and output")
		if not all(isinstance(layer, int) and layer > 0 for layer in layers):
			raise ValueError("layers must only contain int and positive number")
		self.layers = layers
		self.weight = []
		self.biais = []
		self.initialise_value()

	def initialise_value(self):
		print("start init value")
		for i in range(1, len(self.layers)):
			weight_matrix = np.random.randn(self.layers[i], self.layers[i - 1])
			biais_matrix = np.zeros((self.layers[i], 1))
			self.weight.append(weight_matrix)
			self.biais.append(biais_matrix)
	
	def sigmoid(Z):
		print("start Sigmoid")
		return ( 1 / (1 + np.exp(-Z)))

	def Relu(Z):
		print("Start Relu")
		return np.maximum(0, Z)
	
	def forward_propagation(self, data):
		print("start forward propagation")
		for i in range(1, len(self.layers)):

	
	def backward_propagation():
		print("start backward propagation")

	def loss_function():
		print("start loss function")