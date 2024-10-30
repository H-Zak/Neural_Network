import numpy as np

class NewNeuralNetwork():
	def __init__(self, layers): #layers = [1,10,10,1]

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
		self.validation_cost = []
		self.pred_test = []
		self.pred_train = []

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
				A = self.softmax(Z)
			else :
				A = self.sigmoid(Z)
			activation.append(A)

		return activation[-1], Zs, activation



	def softmax(self, z):
		exp_z = np.exp(z - np.max(z))
		return exp_z / np.sum(exp_z, axis=0, keepdims=True)
	
	def backward_propagation(self, X, Y , Zs, activation, alpha):

		m = X.shape[1]
		
		dW = [None] * len(self.weight)
		db = [None] * len(self.biais)
		dA = activation[-1] - Y

		for i in reversed(range(len(self.weight))):
			if i == len(self.weight) - 1:
				dZ = dA 
			else :
				dZ = dA * self.sigmoid_derivative(Zs[i])
			dW[i] = 1/m * np.dot(dZ, activation[i].T) #la derive de A est dans la derive de Z
			
			db[i] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
			if i > 0:
				dA = np.dot(self.weight[i].T, dZ)
			self.weight[i] = self.weight[i] - alpha * dW[i]
			self.biais[i] = self.biais[i] - alpha * db[i]
	
	def prediction_list(self, x_test, y_test, X, Y):
		# Prédictions sur l'ensemble de test
			y_pred_test = self.predict(x_test)
			accuracy_test = np.mean(y_pred_test == y_test) * 100
			self.pred_test.append(accuracy_test)

			# Prédictions sur l'ensemble d'entraînement (optionnel)
			y_pred_train = self.predict(X)
			accuracy_train = np.mean(y_pred_train == Y) * 100
			self.pred_train.append(accuracy_train)

	
	def train(self, X, Y, num_epochs, learning_rate, x_test, y_test):
		if not hasattr(self, 'weight') or not hasattr(self, 'biais'):
			self.initialize_parameters()
		print("Y SHAPE", Y.shape)
		Y = Y.T
		print("Y SHAPE", Y.shape)

		for epoch in range(num_epochs):
			output, Zs, activation = self.forward_propagation(X)

			cost = self.compute_cost(output, Y)
			validation_cost = self.function_valid_cost(x_test, y_test.T)
			self.prediction_list(x_test, y_test.T, X , Y)

			self.backward_propagation(X, Y, Zs, activation, learning_rate)

			if epoch % 100 == 0 :
				self.costs.append(cost)
				self.validation_cost.append(validation_cost)
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
	
	def train_mini_batch(self, X, Y, num_epochs, learning_rate, mini_batch_size,x_test, y_test):
		if not hasattr(self, 'weight') or not hasattr(self, 'biais'):
			self.initialize_parameters()
		
		for epoch in range(num_epochs):
			mini_batches = self.create_mini_batches(X, Y, mini_batch_size)
			for mini_batch in mini_batches:
				(mini_batch_X, mini_batch_Y) = mini_batch
				output, Zs, activation = self.forward_propagation(mini_batch_X)

				cost = self.compute_cost(output, mini_batch_Y)
				validation_cost = self.function_valid_cost(x_test, y_test)
				self.prediction_list(x_test, y_test, X , Y)
				self.backward_propagation(mini_batch_X, mini_batch_Y, Zs, activation, learning_rate)

			if epoch % 100 == 0 :
				self.costs.append(cost)
				self.validation_cost.append(validation_cost)
				print(f"Coût après l'époque {epoch}: {cost}")
		return self.costs
	
	def compute_cost(self, AL, Y):
		# print("Output", AL, "Real OUtput", Y)
		m = Y.shape[1]
		print("AL SHAPE", AL.shape)
		epsilon = 1e-10
		AL = np.clip(AL, epsilon, 1 - epsilon)
		# AL = np.array(AL)  # Ensure AL is a NumPy array of type float64
		# Y = np.array(Y) 
		cost = -1/m * np.sum(Y * np.log(AL))
		cost = np.squeeze(cost)  # Assure que le coût est un scalaire
		return cost
	
	def predict(self, X):
		# print("DATA", X.shape, X)
		activations, _, __ = self.forward_propagation(X)
		predicted_one_hot = np.zeros_like(activations)

		# activation = activations.T
		# print("ACTIVSTION\n", activations, activations.shape)
		predictions = np.argmax(activations, axis=0) # Seuil pour la classification binaire
		predicted_one_hot[predictions, np.arange(predictions.size)] = 1
		# predicted_one_hot = np.zeros_like(activations[-1])
		# predicted_one_hot[predictions, np.arange(predictions.size)] = 1
		print("PREDICTION",predictions, predictions.shape)
		print("Activation",activations, activations.shape)
		print("Predicted one-hot encoding", predicted_one_hot, predicted_one_hot.shape)
		return predicted_one_hot
	
	def function_valid_cost(self, x_test, y_test):
		output, Zs, activation = self.forward_propagation(x_test)
		valid_cost = self.compute_cost(output, y_test)
		return valid_cost



