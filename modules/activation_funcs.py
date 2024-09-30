import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # The derivative of sigmoid function is: σ'(x) = σ(x) * (1 - σ(x))
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # The directive of ReLU function is 1 for x > 0 and 0 for x <= 0
    return np.where(x > 0, 1, 0)

