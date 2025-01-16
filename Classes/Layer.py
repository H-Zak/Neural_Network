import numpy as np
from typing import Tuple, List, Callable
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_shape: int, number_of_neurons: int, layer_type : str):
        self.weights = np.random.rand(number_of_neurons, input_shape) - 0.5
        self.bias = np.zeros((number_of_neurons, 1)) - 0.5
        # Save activation of the previous layer
        self.A_prev = None
        # Save pre-activation
        self.Z = None
        # debug
        self.layer_type = layer_type

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.A_prev = inputs
        self.Z = np.dot(self.weights, inputs) + self.bias
        A = self.activation_ft(self.Z)
        return A

    # TODO -> learning rate here? 
    def update_parameters(self, dW: np.ndarray, db: np.ndarray, learning_rate: float):
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

    @abstractmethod
    def backward(self, dA: np.ndarray, m : int) -> np.ndarray:
        pass
    
    @abstractmethod
    def activation_ft(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative_activation_ft(self, output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

class HiddenLayer(Layer):
    def __init__(self, input_shape: int, number_of_neurons: int, layer_type : str):
        super().__init__(input_shape, number_of_neurons, layer_type)
    
    
    def backward(self, dA: np.ndarray, m : int) -> np.ndarray:
        # Compute dZ
        dZ = dA * self.derivative_activation_ft(self.Z)
        self.dW = 1/m * np.dot(dZ, self.A_prev.T)
        self.db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        # Gradient of the previous layer ?  
        dA_prev = np.dot(self.weights.T, dZ)
        return dA_prev
    
    # reLU function
    def activation_ft(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    # reLU derivative
    def derivative_activation_ft(self, output: np.ndarray) -> np.ndarray:
        return np.where(output > 0, 1, 0)

    def get_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dW, self.db


class OutputLayer(Layer):
    def __init__(self, input_shape: int, number_of_neurons: int, layer_type : str):
        super().__init__(input_shape, number_of_neurons, layer_type)

    def backward(self, dA: np.ndarray, m : int) -> np.ndarray:
        dZ = dA
        self.dW = 1 / m * np.dot(dZ, self.A_prev.T)
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.weights.T, dZ)
        return dA_prev

    def activation_ft(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative_activation_ft(self, output: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return output - y_true

    def get_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dW, self.db 


# import numpy as np
# from typing import List, Callable
# from abc import ABC, abstractmethod

# class Layer(ABC):
#     @abstractmethod
#     def __init__(self, input_shape, number_of_neurons, layer_type : str):
#         self.weights = np.random.randn(number_of_neurons, input_shape)
#         self.bias = np.zeros((number_of_neurons, 1))

#     def call(self, inputs):
#         Z = np.dot(self.weights, inputs) + self.bias
#         A = self.activation_ft(Z)
#         return A, Z

#     @abstractmethod
#     def backward(self, dA):
#         # Implementar la lógica de retropropagación
#         # Por ejemplo, calcula dZ y almacena dW y db para usar en get_gradients
#         pass

#     @abstractmethod
#     def get_gradients(self):
#         # Devuelve dW y db almacenados durante la retropropagación
#         return self.dW, self.db

#     def update_parameters(self, dW, db, learning_rate=0.01):
#         self.weights -= learning_rate * dW
#         self.bias -= learning_rate * db
    
#     @abstractmethod
#     def activation_ft(self, x):
#         pass

#     @abstractmethod
#     def derivative_activation_ft(self, output):
#         pass

# class HiddenLayer(Layer):
#     def __init__(self, input_shape, number_of_neurons, layer_type : str):
#         super().__init__(input_shape, number_of_neurons, layer_type)
#         self.type_layer = "Hidden Layer"
        
#     # def call(self, inputs):
#     #     Z = np.dot(self.weights, inputs) + self.bias
#     #     # print(f"------------ Zs  hidden layer ------------------------")
#     #     # print(Z)
#     #     A = self.activation_ft(Z)
#     #     # print(f"----------- Activations  Hidden Layer ----------------")
#     #     # print(A)
#     #     return A, Z

#     # reLU function
#     def activation_ft(self, x):
#         return np.maximum(0, x)

#     # reLU derivative
#     def derivative_activation_ft(self, x):
#         # The directive of ReLU function is 1 for x > 0 and 0 for x <= 0
#         return np.where(x > 0, 1, 0)

# class OutputLayer(Layer):
#     def __init__(self, input_shape, number_of_neurons, layer_type : str):
#         super().__init__(input_shape, number_of_neurons, layer_type)
#         self.type_layer = "Output Layer"

#     # def call(self, inputs):
#     #     # print(inputs)
#     #     # print(self.weights)
#     #     Z = np.dot(self.weights, inputs) + self.bias
#     #     # print(f"------------ Zs  Output layer ------------------------")
#     #     # print(Z)
#     #     # print(Z.shape)
#     #     A = self.activation_ft(Z)
#     #     # print(f"----------- Activations  Output layer ----------------")
#     #     # print(A)
#     #     # print(A.shape)
#     #     return A, Z

#     # Softmax
#     def activation_ft(self, x):
#         exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#         return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
#     def derivative_activation_ft(self, output, y_true):
#         # y_true is a one-hot encoded vector
#         return output - y_true
    
#     # # Sigmoid
#     # def activation_ft(self, x):
#     #     return 1 / (1 + np.exp(-x))

#     # def derivative_activation_ft(self, output):
#     #     # The derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
#     #     sig = self.activation_ft(output)
#     #     return sig * (1 - sig)