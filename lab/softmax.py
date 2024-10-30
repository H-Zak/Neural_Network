import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Estabilidad numérica
    return exp_logits / np.sum(exp_logits)

# Logits de la capa de salida
logits = np.array([0.23, 3.25])

# Aplicar softmax para obtener probabilidades
probabilidades = softmax(logits)
print("Probabilidad de la clase 0:", probabilidades[0])
print("Probabilidad de la clase 1:", probabilidades[1])



def backpropagation(self, inputs, outputs, Y, learning_rate):
    m = inputs.shape[1]  # Número de muestras en el batch
    
    # Derivative arrays
    dW = [None] * len(self.layers)
    db = [None] * len(self.layers)

    # Derivative of cross-entropy loss function for softmax (dA = softmax(outputs) - Y)
    dA = outputs - Y  # Este cálculo es suficiente para la última capa
    
    # Backpropagation loop (iterate over layers in reverse order)
    for idx in reversed(range(0, len(self.layers))):
        # Si es la capa de salida (softmax), dA ya es la derivada
        if idx == len(self.layers) - 1:
            dZ = dA  # Para la capa de salida, dZ ya es la diferencia (softmax - Y)
        else:
            # Para capas ocultas, multiplicamos por la derivada de la función de activación
            dZ = dA * self.layers[idx].derivative_activation_ft(self.Zs[idx])  # Zs = preactivaciones
        
        # Compute gradient with respect to the weights (dW)
        dW[idx] = (1/m) * np.dot(dZ, self.activations[idx].T)  # activations[idx] es la activación anterior
        # Compute gradient with respect to the bias (db)
        db[idx] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Update weights and biases
        self.layers[idx].weights -= learning_rate * dW[idx]
        self.layers[idx].bias -= learning_rate * db[idx]
        
        # Propagate dA for the next layer (except for the last layer)
        if idx > 0:
            dA = np.dot(self.layers[idx].weights.T, dZ)
