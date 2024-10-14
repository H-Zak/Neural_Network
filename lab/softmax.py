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
