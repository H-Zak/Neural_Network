from sklearn.model_selection import train_test_split
import pandas as pd

# Classes imports
from Classes.DatasetLoader import DatasetLoader
from Classes.Model import Model
from Classes.NeuralNetwork import NeuralNetwork
# Loss function
from modules.loss_function import binary_cross_entropy

def main():
	try:
		data_loader = DatasetLoader("./dataset/data_cancer.csv")

		neural_network = NeuralNetwork(input_shape=data_loader.get_input_shape(), hidden_layers=[24, 24], output_shape=1)

		model = Model(network=neural_network, 
                      data_train=data_loader.get_train_data(),
                      data_valid=data_loader.get_test_data(),
                      loss_function=binary_cross_entropy, 
                      learning_rate=0.1, 
                      batch_size=16,
                      epochs=1000)
		# Start the training
		model.train()
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print('Failed to read the dataset')

if __name__ == "__main__":
	main()