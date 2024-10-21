from sklearn.model_selection import train_test_split
import pandas as pd

# Classes imports
from Classes.DatasetLoader import DatasetLoader
from Classes.Model import Model
from Classes.NeuralNetwork import NeuralNetwork
# Loss function
# from modules.loss_function import binary_cross_entropy, cross_entropy_loss
from modules.loss_function import cross_entropy_loss

def main():
	try:
		data_loader = DatasetLoader("./dataset/data_test.csv")

		neural_network = NeuralNetwork(input_shape=data_loader.get_input_shape(), hidden_layers=[24, 24], output_shape=2)

		model = Model(network=neural_network, 
                      data_train=data_loader.get_train_data(),
                      data_valid=data_loader.get_test_data(),
                      loss_function=cross_entropy_loss, 
                      learning_rate=0.23, 
                      batch_size=8,
                      epochs=10)
		# Start the training
		model.train()
	# except ValueError as e:
	# 	print(e)
	except FileNotFoundError:
		print('Failed to read the dataset')

if __name__ == "__main__":
	main()