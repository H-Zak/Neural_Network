import numpy as np
from typing import List, Callable
from Classes.NeuralNetwork import NeuralNetwork

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
        # loss_function (Callable): Loss function to be used during training.
        learning_rate (float): Learning rate for optimization. Default is 0.001.
        batch_size (int): Number of training examples per batch. Default is 2.
        epochs (int): Number of training iterations over the entire dataset. Default is 5000.
        """
        
        # Initialize the neural network
        self.network = network
        
        # Training and validation data
        self.x_train = data_train[0]
        self.y_train = data_train[1]

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

    # def train(self):
    #     inputs =  self.x_train
    #     tolerance : float = 1e-8
    #     for e in range(self.epochs):
    #         outputs = self.network.feedforward(inputs)
    #         cost = self.loss_function(self.y_train, outputs)
    #         if e % 100 == 0:
    #             self.train_loss_history.append(cost)
    #         print(f"epoch {e+1}/{self.epochs} - loss: {cost:2.4f} - val_loss: {0}")
    #         self.network.backpropagation(inputs, outputs, self.y_train, self.learning_rate)

    def train(self):
        inputs = self.x_train

        print(inputs.shape)
        tolerance: float = 1e-8
        count_training_examples = inputs.shape[1]
        
        for e in range(self.epochs):
            indices = np.random.permutation(count_training_examples)
            for start in range(0, count_training_examples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                x_batch = self.x_train[:, batch_indices]
                y_batch = self.y_train[:, batch_indices]

                # print(x_batch)

                # print(y_batch)

                outputs = self.network.feedforward(x_batch)

                cost = self.loss_function(y_batch, outputs)

                self.network.backpropagation(x_batch, outputs, y_batch, self.learning_rate)

                if e % 100 == 0 and e == 0:
                    self.train_loss_history.append(cost)

            print(f"epoch {e+1}/{self.epochs} - loss: {cost:2.4f} - val_loss: {0}")