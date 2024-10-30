import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, A_prev):
        """
        Propagation avant pour une couche.
        :param A_prev: Entrées de la couche précédente (ou données d'entrée pour la première couche)
        :return: Z et A (sortie activée)
        """
        Z = np.dot(self.weights, A_prev) + self.biases
        A = self.activation(Z)
        return Z, A

    def backward(self, dA, Z, A_prev, m):
        """
        Rétropropagation pour une couche.
        :param dA: Gradient de la sortie activée
        :param Z: Valeur Z de la couche
        :param A_prev: Entrée de la couche précédente
        :param m: Nombre d'exemples
        :return: Gradient dW, db, et dA_prev pour la couche précédente
        """
        dZ = dA * self.activation_derivative(Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)
        return dW, db, dA_prev

    def update_parameters(self, dW, db, learning_rate):
        """
        Mise à jour des paramètres de la couche (poids et biais).
        """
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
