import numpy as np
import os


class MultilayerPerceptron():
	def __init__(self, layers):
		if not isinstance(layers, (tuple, list)):
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
		self.pred_val = []
		self.pred_train = []

	def initialise_value(self):
		for i in range(1, len(self.layers)):
			# Xavier initialization pour sigmoid
			scale = np.sqrt(1 / self.layers[i - 1])
			weight_matrix = np.random.randn(self.layers[i], self.layers[i - 1]) * scale
			biais_matrix = np.zeros((self.layers[i], 1))
			self.weight.append(weight_matrix)
			self.biais.append(biais_matrix)

	def save_weights(self, filepath="Result/model_weights.npz"):
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		np.savez(filepath,
			weights=np.array(self.weight, dtype=object),
			biais=np.array(self.biais, dtype=object),
			layers=np.array(self.layers))
		print(f"Poids sauvegardés dans {filepath}")

	def load_weights(self, filepath="Result/model_weights.npz"):
		data = np.load(filepath, allow_pickle=True)
		loaded_layers = data['layers']
		if not np.array_equal(loaded_layers, self.layers):
			raise ValueError(
				f"L'architecture chargée ({loaded_layers}) "
				f"ne correspond pas à l'architecture actuelle ({self.layers}).")
		self.weight = list(data['weights'])
		self.biais = list(data['biais'])
		if len(self.weight) != len(self.biais) or len(self.weight) != len(self.layers) - 1:
			raise ValueError("Incohérence dans les dimensions des poids/biais chargés.")
		for i in range(len(self.weight)):
			expected_w = (self.layers[i + 1], self.layers[i])
			expected_b = (self.layers[i + 1], 1)
			if self.weight[i].shape != expected_w or self.biais[i].shape != expected_b:
				raise ValueError(
					f"Dimension incorrecte couche {i + 1}. "
					f"Attendu W:{expected_w} b:{expected_b}. "
					f"Trouvé W:{self.weight[i].shape} b:{self.biais[i].shape}")
		print(f"Poids chargés depuis {filepath}")

	# --- Fonctions d'activation ---

	def sigmoid(self, Z):
		return 1 / (1 + np.exp(-Z))

	def sigmoid_derivative(self, Z):
		s = self.sigmoid(Z)
		return s * (1 - s)

	def relu(self, Z):
		return np.maximum(0, Z)

	def relu_derivative(self, Z):
		return np.where(Z > 0, 1, 0)

	def softmax(self, z):
		exp_z = np.exp(z - np.max(z))
		return exp_z / np.sum(exp_z, axis=0, keepdims=True)

	# --- Propagation ---

	def forward_propagation(self, data):
		A = data
		activation = [A]
		Zs = []

		for i in range(len(self.weight)):
			Z = np.dot(self.weight[i], A) + self.biais[i]
			Zs.append(Z)
			if i == len(self.weight) - 1:
				A = self.softmax(Z)
			else:
				A = self.sigmoid(Z)
			activation.append(A)

		return activation[-1], Zs, activation

	def backward_propagation(self, X, Y, Zs, activation, alpha):
		m = X.shape[1]
		dA = activation[-1] - Y

		for i in reversed(range(len(self.weight))):
			if i == len(self.weight) - 1:
				dZ = dA
			else:
				dZ = dA * self.sigmoid_derivative(Zs[i])

			dW = (1 / m) * np.dot(dZ, activation[i].T)
			db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

			if i > 0:
				dA = np.dot(self.weight[i].T, dZ)

			self.weight[i] -= alpha * dW
			self.biais[i] -= alpha * db

	# --- Coût ---

	def compute_cost(self, AL, Y):
		m = Y.shape[1]
		epsilon = 1e-10
		AL = np.clip(AL, epsilon, 1 - epsilon)
		cost = -1 / m * np.sum(Y * np.log(AL))
		return np.squeeze(cost)

	def function_valid_cost(self, x_val, y_val):
		output, _, _ = self.forward_propagation(x_val)
		return self.compute_cost(output, y_val)

	# --- Prédiction ---

	def predict(self, X):
		activations, _, _ = self.forward_propagation(X)
		predicted_one_hot = np.zeros_like(activations)
		predictions = np.argmax(activations, axis=0)
		predicted_one_hot[predictions, np.arange(predictions.size)] = 1
		return predicted_one_hot

	def prediction_list(self, x_val, y_val_T, X_train, Y_train_T):
		# Validation
		y_pred_val = self.predict(x_val)
		true_labels_val = np.argmax(y_val_T, axis=0)
		pred_labels_val = np.argmax(y_pred_val, axis=0)
		self.pred_val.append(np.mean(pred_labels_val == true_labels_val) * 100)

		# Train
		y_pred_train = self.predict(X_train)
		true_labels_train = np.argmax(Y_train_T, axis=0)
		pred_labels_train = np.argmax(y_pred_train, axis=0)
		self.pred_train.append(np.mean(pred_labels_train == true_labels_train) * 100)

	# --- Entraînement ---

	def train(self, X, Y, num_epochs, learning_rate, x_val, y_val, patience=0):
		Y = Y.T
		best_val_cost = float('inf')
		patience_counter = 0
		best_weights = None
		best_biais = None

		for epoch in range(num_epochs):
			output, Zs, activation = self.forward_propagation(X)
			self.backward_propagation(X, Y, Zs, activation, learning_rate)

			if epoch % 100 == 0:
				cost = self.compute_cost(output, Y)
				val_cost = self.function_valid_cost(x_val, y_val.T)
				self.prediction_list(x_val, y_val.T, X, Y)
				self.costs.append(cost)
				self.validation_cost.append(val_cost)
				print(f"Époque {epoch}: coût = {cost:.6f}, val_coût = {val_cost:.6f}")

				if patience > 0:
					if val_cost < best_val_cost:
						best_val_cost = val_cost
						patience_counter = 0
						best_weights = [w.copy() for w in self.weight]
						best_biais = [b.copy() for b in self.biais]
					else:
						patience_counter += 1
						if patience_counter >= patience:
							print(f"Early stopping à l'époque {epoch} (patience={patience})")
							self.weight = best_weights
							self.biais = best_biais
							break

		return self.costs

	def create_mini_batches(self, X, Y, mini_batch_size):
		m = X.shape[1]
		mini_batches = []
		permutation = np.random.permutation(m)
		X_shuffled = X[:, permutation]
		Y_shuffled = Y[:, permutation]

		num_complete = m // mini_batch_size
		for k in range(num_complete):
			mini_batches.append((
				X_shuffled[:, k * mini_batch_size:(k + 1) * mini_batch_size],
				Y_shuffled[:, k * mini_batch_size:(k + 1) * mini_batch_size]
			))

		if m % mini_batch_size != 0:
			mini_batches.append((
				X_shuffled[:, num_complete * mini_batch_size:],
				Y_shuffled[:, num_complete * mini_batch_size:]
			))

		return mini_batches

	def train_mini_batch(self, X, Y, num_epochs, learning_rate, mini_batch_size, x_val, y_val, patience=0):
		Y = Y.T
		y_val_T = y_val.T
		best_val_cost = float('inf')
		patience_counter = 0
		best_weights = None
		best_biais = None

		for epoch in range(num_epochs):
			mini_batches = self.create_mini_batches(X, Y, mini_batch_size)
			for mini_batch_X, mini_batch_Y in mini_batches:
				output, Zs, activation = self.forward_propagation(mini_batch_X)
				self.backward_propagation(mini_batch_X, mini_batch_Y, Zs, activation, learning_rate)

			if epoch % 100 == 0:
				full_output, _, _ = self.forward_propagation(X)
				cost = self.compute_cost(full_output, Y)
				val_cost = self.function_valid_cost(x_val, y_val_T)
				self.prediction_list(x_val, y_val_T, X, Y)
				self.costs.append(cost)
				self.validation_cost.append(val_cost)
				print(f"Époque {epoch}: coût = {cost:.6f}, val_coût = {val_cost:.6f}")

				if patience > 0:
					if val_cost < best_val_cost:
						best_val_cost = val_cost
						patience_counter = 0
						best_weights = [w.copy() for w in self.weight]
						best_biais = [b.copy() for b in self.biais]
					else:
						patience_counter += 1
						if patience_counter >= patience:
							print(f"Early stopping à l'époque {epoch} (patience={patience})")
							self.weight = best_weights
							self.biais = best_biais
							break

		return self.costs
