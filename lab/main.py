from sklearn.model_selection import train_test_split
import pandas as pd

# Classes imports
from Classes.Model import Model
from Classes.NeuralNetwork import NeuralNetwork
# Loss function
from modules.loss_function import binary_cross_entropy


def main():
	try:
		data =  pd.read_csv("./dataset/data_test.csv", header=None)

		# Splitting data manually
		x_train = data.iloc[:,2:4].T
		y_train = data.iloc[:, 1].map({'M' : 1, 'B' : 0}).to_numpy()

		# print(x)
		# print(y)

		# x_train = x.iloc[[0, 1]]
		# y_train = y.iloc[[0, 1]]

		# x_remaining = x.drop(x_train.index)
		# y_remaining = y.drop(y_train.index)

		# x_test = x_remaining.iloc[:2]
		# y_test = y_remaining.iloc[:2]

		# print("---- Initial input shape -------")
		# print(x_train)
		# print("--------------------------")
		# print("---- Initial input shape -------")
		# print(x_train.shape)
		# print("--------------------------")

		# print(y_train.shape)
		neural_network = NeuralNetwork(input_shape=x_train.shape[0], hidden_layers=[24, 24, 24], output_shape=1)
		# Initialize model
		model = Model(network=neural_network, 
                      data_train=(x_train, y_train),
                      data_valid=([], []),
                      loss_function=binary_cross_entropy, 
                      learning_rate=0.1, 
                      batch_size=2,
                      epochs=5000)
		# Start the training
		model.train()
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print('Failed to read the dataset')

if __name__ == "__main__":
	main()