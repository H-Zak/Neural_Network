from sklearn.model_selection import train_test_split
import pandas as pd

# Classes imports
from Classes.Model import Model
from Classes.NeuralNetwork import NeuralNetwork
# Activation functions imports
from modules.activation_funcs import sigmoid, relu
# Loss function
from modules.loss_function import binary_cross_entropy

def main():
	try:
		data =  pd.read_csv("./dataset/data_test.csv", header=None)

		# Splitting data manually
		x = data.drop(data.columns[1], axis=1)
		x = x.iloc[:,1:]
		y = data.iloc[:, 1]

		x_train = x.iloc[[0, 1]]
		y_train = y.iloc[[0, 1]]

		x_remaining = x.drop(x_train.index)
		y_remaining = y.drop(y_train.index)

		x_test = x_remaining.iloc[:2]
		y_test = y_remaining.iloc[:2]

		# # Splitting data
		# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
		# shape[1] -> Number of characteristics for each example
		neural_network = NeuralNetwork(x_train.shape[1], [2, 1], sigmoid)

		# Initialize model
		model = Model(network=neural_network, 
                      data_train=(x_train, y_train),
                      data_valid=(x_test, y_test),
                      loss_function=binary_cross_entropy, 
                      learning_rate=0.001, 
                      batch_size=2,
                      epochs=1)
		# Start the training
		model.train()
		#test with y_test si les resultat sont bons
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print('Failed to read the dataset')
	


if __name__ == "__main__":
	main()