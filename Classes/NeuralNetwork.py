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
		self.costs = []

	def initialise_value(self):

		for i in range(1, len(self.layers)):
			weight_matrix = np.random.randn(self.layers[i], self.layers[i - 1])
			biais_matrix = np.zeros((self.layers[i], 1))
			self.weight.append(weight_matrix)
			self.biais.append(biais_matrix)
	
	def sigmoid(self, Z):

		return ( 1 / (1 + np.exp(-Z)))

	def Relu(self, Z):

		return np.maximum(0, Z)

	def sigmoid_derivative(self, Z):
		s = 1 / (1 + np.exp(-Z))
		return s * (1 - s)
	
	def relu_derivative(self, Z):
		return np.where(Z > 0, 1, 0)
	def forward_propagation(self, data):

		A = data
		activation = [A]
		Zs=[]

		for i in range(len(self.weight)):

			Z = np.dot(self.weight[i], A) + self.biais[i]
			Zs.append(Z)
			if i == len(self.weight) - 1 :
				A = self.sigmoid(Z)
			else :
				A = self.Relu(Z)
			activation.append(A)

		return activation[-1], Zs, activation




	
	def backward_propagation(self, X, Y , Zs, activation, alpha):

		m = X.shape[1]
		dW = [None] * len(self.weight)
		db = [None] * len(self.biais)
		dA = activation[-1] - Y 
		for i in reversed(range(len(self.weight))):
			if i == len(self.weight) - 1:
				dZ = dA * self.sigmoid_derivative(Zs[i])
			else :
				dZ = dA * self.relu_derivative(Zs[i]) # faire la fonction de la derivative de relu
			dW[i] = 1/m * np.dot(dZ, activation[i].T) #la derive de A est dans la derive de Z
			db[i] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
			if i > 0:
				dA = np.dot(self.weight[i].T, dZ)
			self.weight[i] = self.weight[i] - alpha * dW[i]
			self.biais[i] = self.biais[i] - alpha * db[i]
	
	def train(self, X, Y, num_epochs, learning_rate):
		if not hasattr(self, 'weight') or not hasattr(self, 'biais'):
			self.initialize_parameters()
		
		for epoch in range(num_epochs):
			output, Zs, activation = self.forward_propagation(X)

			cost = self.compute_cost(output, Y)
			self.backward_propagation(X, Y, Zs, activation, learning_rate)

			if epoch % 100 == 0 :
				self.costs.append(cost)
				print(f"Coût après l'époque {epoch}: {cost}")
		return self.costs
	
	def create_mini_batches(self, X, Y, mini_batch_size):
		m = X.shape[1]
		mini_batches = []
		permutation = np.random.permutation(m)
		X_shuffled = X[:, permutation]
		Y_shuffled = Y[:, permutation]

		num_complete_minibatches = m // mini_batch_size
		for k in range(num_complete_minibatches):
			mini_batch_X = X_shuffled[:, k * mini_batch_size:(k + 1) * mini_batch_size]
			mini_batch_Y = Y_shuffled[:, k * mini_batch_size:(k + 1) * mini_batch_size]
			mini_batches.append((mini_batch_X, mini_batch_Y))

		if m % mini_batch_size != 0:
			mini_batch_X = X_shuffled[:, num_complete_minibatches * mini_batch_size:]
			mini_batch_Y = Y_shuffled[:, num_complete_minibatches * mini_batch_size:]
			mini_batches.append((mini_batch_X, mini_batch_Y))

		return mini_batches
	
	def train_mini_batch(self, X, Y, num_epochs, learning_rate, mini_batch_size):
		if not hasattr(self, 'weight') or not hasattr(self, 'biais'):
			self.initialize_parameters()
		
		for epoch in range(num_epochs):
			mini_batches = self.create_mini_batches(X, Y, mini_batch_size)
			for mini_batch in mini_batches:
				(mini_batch_X, mini_batch_Y) = mini_batch
				output, Zs, activation = self.forward_propagation(mini_batch_X)

				cost = self.compute_cost(output, mini_batch_Y)
				self.backward_propagation(mini_batch_X, mini_batch_Y, Zs, activation, learning_rate)

			if epoch % 100 == 0 :
				self.costs.append(cost)
				print(f"Coût après l'époque {epoch}: {cost}")
		return self.costs
	
	def compute_cost(self, AL, Y):
		m = Y.shape[1]
		epsilon = 1e-10
		AL = np.clip(AL, epsilon, 1 - epsilon)
		# AL = np.array(AL)  # Ensure AL is a NumPy array of type float64
		# Y = np.array(Y) 
		cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
		cost = np.squeeze(cost)  # Assure que le coût est un scalaire
		return cost
	
	def predict(self, X):
		activations, _, __ = self.forward_propagation(X)
		predictions = activations[-1] > 0.5  # Seuil pour la classification binaire
		return predictions.astype(int)

