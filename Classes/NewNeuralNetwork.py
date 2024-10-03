import Classes.layers

class NeuralNetwork:
    def __init__(self, layer_dims, activations, activations_derivatives):
        """
        :param layer_dims: Liste contenant le nombre de neurones par couche [n_x, n_h1, ..., n_y]
        :param activations: Liste des fonctions d'activation pour chaque couche
        :param activations_derivatives: Liste des dérivées des fonctions d'activation
        """
        self.layers = []
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(layer_dims[i - 1], layer_dims[i], activations[i - 1], activations_derivatives[i - 1]))

    def forward_propagation(self, X):
        A = X
        activations = [A]
        Zs = []
        for layer in self.layers:
            Z, A = layer.forward(A)
            Zs.append(Z)
            activations.append(A)
        return Zs, activations

    def backward_propagation(self, Y, Zs, activations, learning_rate):
        m = Y.shape[1]
        dA = activations[-1] - Y
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dW, db, dA = layer.backward(dA, Zs[i], activations[i], m)
            layer.update_parameters(dW, db, learning_rate)

    def train(self, X, Y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            Zs, activations = self.forward_propagation(X)
            self.backward_propagation(Y, Zs, activations, learning_rate)
