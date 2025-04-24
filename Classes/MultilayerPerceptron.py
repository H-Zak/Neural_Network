import numpy as np
import os

class MultilayerPerceptron():
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
		self.pred_val = []
		self.pred_train = []

	def save_weights(self, filepath="Result/model_weights.npz"):
		# """Sauvegarde les poids et les biais du réseau dans un fichier."""
		try:
			# Crée le dossier Result s'il n'existe pas
			os.makedirs(os.path.dirname(filepath), exist_ok=True)
			# Utilise np.savez pour sauvegarder plusieurs arrays dans un seul fichier .npz
			# Convertit les listes de matrices en arrays d'objets pour la sauvegarde
			np.savez(filepath, weights=np.array(self.weight, dtype=object), biais=np.array(self.biais, dtype=object), layers=np.array(self.layers))
			print(f"Poids sauvegardés avec succès dans {filepath}")
		except Exception as e:
			print(f"Erreur lors de la sauvegarde des poids : {e}")

	def load_weights(self, filepath="Result/model_weights.npz"):
	# """Charge les poids et les biais depuis un fichier."""
		try:
			data = np.load(filepath, allow_pickle=True)
			# Vérifie si l'architecture chargée correspond à l'architecture actuelle
			loaded_layers = data['layers']
			if not np.array_equal(loaded_layers, self.layers):
				raise ValueError(f"L'architecture du modèle chargé ({loaded_layers}) ne correspond pas à l'architecture actuelle ({self.layers}).")

			self.weight = list(data['weights'])
			self.biais = list(data['biais'])
			print(f"Poids chargés avec succès depuis {filepath}")
			# Valider les dimensions (optionnel mais recommandé)
			if len(self.weight) != len(self.biais) or len(self.weight) != len(self.layers) - 1:
					raise ValueError("Incohérence dans les dimensions des poids/biais chargés.")
			for i in range(len(self.weight)):
				expected_w_shape = (self.layers[i+1], self.layers[i])
				expected_b_shape = (self.layers[i+1], 1)
				if self.weight[i].shape != expected_w_shape or self.biais[i].shape != expected_b_shape:
					raise ValueError(f"Dimension incorrecte pour la couche {i+1}. Attendu Poids: {expected_w_shape}, Biais: {expected_b_shape}. Trouvé Poids: {self.weight[i].shape}, Biais: {self.biais[i].shape}")

		except FileNotFoundError:
			print(f"Erreur : Fichier de poids '{filepath}' non trouvé.")
			# Optionnel : Initialiser les poids si le fichier n'est pas trouvé ?
			# self.initialise_value()
			raise
		except ValueError as ve: # Attraper spécifiquement ValueError (architecture ou dimensions)
			print(f"Erreur de chargement (ValueError): {ve}")
			raise # Relancer pour que le test puisse l'attraper
		except Exception as e:
			print(f"Erreur lors du chargement des poids : {e}")
			# Optionnel: Re-initialiser en cas d'erreur ?
			# self.initialise_value()
			raise

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

	# Dans Amelioration_2/Classes/MultilayerPerceptron.py

# Dans Amelioration_2/Classes/MultilayerPerceptron.py

	def prediction_list(self, x_val, y_val_transposed, X_train, Y_train_transposed):
		# Note: Renommer les paramètres pour indiquer qu'ils sont transposés est une bonne idée

		# === Validation Set ===
		y_pred_val = self.predict(x_val) # Shape (2, N_val)

		# --- CORRECTION VÉRIFICATION y_val ---
		# Vérifier la forme de y_val_transposed (attendu: (2, N_val))
		if y_val_transposed.shape[0] != 2:
			raise ValueError(f"Erreur dans prediction_list: La forme de y_val_transposed ({y_val_transposed.shape}) n'est pas (2, N).")
		# Obtenir les vrais labels depuis y_val_transposed (one-hot, shape (2, N))
		true_labels_val = np.argmax(y_val_transposed, axis=0) # L'argmax est sur l'axe des classes (axe 0)

		if y_pred_val.shape[0] != 2:
			raise ValueError(f"Erreur dans prediction_list: La forme de y_pred_val ({y_pred_val.shape}) n'est pas (2, N).")
		pred_labels_val = np.argmax(y_pred_val, axis=0)
		accuracy_val = np.mean(pred_labels_val == true_labels_val) * 100
		self.pred_val.append(accuracy_val) # Si renommé

		# === Training Set ===
		y_pred_train = self.predict(X_train) # Shape (2, N_train)

		# --- CORRECTION VÉRIFICATION Y_train ---
		# Vérifier la forme de Y_train_transposed (attendu: (2, N_train))
		if Y_train_transposed.shape[0] != 2:
			raise ValueError(f"Erreur dans prediction_list: La forme de Y_train_transposed ({Y_train_transposed.shape}) n'est pas (2, N).")
		# Obtenir les vrais labels depuis Y_train_transposed (one-hot, shape (2, N))
		true_labels_train = np.argmax(Y_train_transposed, axis=0) # L'argmax est sur l'axe des classes (axe 0)

		if y_pred_train.shape[0] != 2:
			raise ValueError(f"Erreur dans prediction_list: La forme de y_pred_train ({y_pred_train.shape}) n'est pas (2, N).")
		pred_labels_train = np.argmax(y_pred_train, axis=0)
		accuracy_train = np.mean(pred_labels_train == true_labels_train) * 100
		self.pred_train.append(accuracy_train)

	def train(self, X, Y, num_epochs, learning_rate, x_val, y_val):
		if not hasattr(self, 'weight') or not hasattr(self, 'biais'):
			self.initialize_parameters()
		print("Y SHAPE", Y.shape)
		Y = Y.T
		print("Y SHAPE", Y.shape)

		for epoch in range(num_epochs):
			output, Zs, activation = self.forward_propagation(X)

			cost = self.compute_cost(output, Y)
			validation_cost = self.function_valid_cost(x_val, y_val.T)
			self.prediction_list(x_val, y_val.T, X , Y)

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

	def train_mini_batch(self, X, Y, num_epochs, learning_rate, mini_batch_size,x_val, y_val):
		if not hasattr(self, 'weight') or not hasattr(self, 'biais'):
			self.initialize_parameters()

		for epoch in range(num_epochs):
			mini_batches = self.create_mini_batches(X, Y, mini_batch_size)
			for mini_batch in mini_batches:
				(mini_batch_X, mini_batch_Y) = mini_batch
				output, Zs, activation = self.forward_propagation(mini_batch_X)

				cost = self.compute_cost(output, mini_batch_Y)
				validation_cost = self.function_valid_cost(x_val, y_val)
				self.prediction_list(x_val, y_val, X , Y)
				self.backward_propagation(mini_batch_X, mini_batch_Y, Zs, activation, learning_rate)

			if epoch % 100 == 0 :
				self.costs.append(cost)
				self.validation_cost.append(validation_cost)
				print(f"Coût après l'époque {epoch}: {cost}")
		return self.costs

	def compute_cost(self, AL, Y):
		# print("Output", AL, "Real OUtput", Y)
		m = Y.shape[1]
		# print("AL SHAPE", AL.shape)
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
		# print("PREDICTION",predictions, predictions.shape)
		# print("Activation",activations, activations.shape)
		# print("Predicted one-hot encoding", predicted_one_hot, predicted_one_hot.shape)
		return predicted_one_hot

	def function_valid_cost(self, x_val, y_val):
		output, Zs, activation = self.forward_propagation(x_val)
		valid_cost = self.compute_cost(output, y_val)
		return valid_cost



