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
		A = data
		activation = [A]
		Zs=[]

		for i in range(len(self.weight)):
			# Z = self.linear_combinaison(self.weight[i], A, self.biais[i])
			Z = np.dot(self.weight[i], A) + self.biais
			Zs.append(Z)
			if i == len(self.weight) - 1 :
				A = self.sigmoid(Z)
			else :
				A = self.Relu(Z)
			activation.append(A)
		return activation[-1], Zs, activation




	
	def backward_propagation(self, X, Y , Zs, activation):
		print("start backward propagation")
		m = X.shape[1]
		dW = [None] * len(self.weight)
		db = [None] * len(self.biais)
		dA = activation[-1] - Y 
		for i in reversed(range(len(self.weight))):
			if i == len(self.weight):
				dZ = dA * self.sigmoid_derivative(Zs[i]) #il faut coder la derivative
			else :
				dZ = dA * self.relu_derivative(Zs[i]) # faire la fonction de la derivative de relu
			dW[i] = 1/m * np.dot(dZ, activation[i].T)
			db[i] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
			dA = np.dot(self.weight.T, dZ)
	
	
	def loss_function():
		print("start loss function")