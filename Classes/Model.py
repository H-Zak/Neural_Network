import numpy as np
from typing import List, Callable
from Classes.NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from modules.loss_function import binary_cross_entropy

class Model():
    def __init__(self,
                network: NeuralNetwork,
                data_train: tuple,  # (x_train, y_train)
                data_valid: tuple,  # (x_valid, y_valid)
                loss_function: Callable,
                learning_rate: float = 0.001,
                batch_size: int = 2,
                epochs: int = 5000):
        """
        Constructor for the Model class.
        
        Parameters:
        network (NeuralNetwork): The neural network architecture.
        data_train (np.ndarray): Training data as a numpy array.
        data_valid (np.ndarray): Validation data as a numpy array.
        loss_function (Callable): Loss function to be used during training.
        learning_rate (float): Learning rate for optimization. Default is 0.001.
        batch_size (int): Number of training examples per batch. Default is 2.
        epochs (int): Number of training iterations over the entire dataset. Default is 5000.
        """
        
        # Initialize the neural network
        self.network = network
        
        # Training and validation data
        self.data_train = data_train
        self.data_valid = data_valid
        
        # Loss function and learning rate
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        
        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Keep track of loss during training and validation
        self.train_loss_history = []
        self.valid_loss_history = []

    def scaling_inputs(self):
        inputs = self.data_train[0].to_numpy()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(inputs)
        return x_scaled

    def train(self):
        inputs = self.scaling_inputs()
        for e in range(self.epochs):
            # Outputs of softmax function
            outputs = self.network.feedforward(inputs)
            correct_class_probs = outputs[range(len(outputs)), self.data_train[1]]
            cost = self.loss_function(self.data_train[1], correct_class_probs)
            print(f"epoch {e+1}/{self.epochs} - loss: {cost} - val_loss: {0}")
            self.network.backpropagation(outputs, self.data_train[1])
        
        # print("-------------Zs-----------------")
        # print(self.network.Zs)
        # print("------------- Activations -----------------")
        # print(self.network.activations)