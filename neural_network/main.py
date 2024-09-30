from sklearn.model_selection import train_test_split
import pandas as pd

# Classes imports
from Classes.NeuralNetwork import NeuralNetwork
# Activation functions imports
from modules.activation_funcs import sigmoid

def main():
	try:
		data =  pd.read_csv("./dataset/data_cancer.csv")
		x = data.drop(data.columns[1], axis=1)
		y = data.iloc[:, 1]
		# Splitting data
		x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
		
		# Start the creation of the neural
		NeuralNetwork(x_train, [24, 24], sigmoid)
		# Start the training 

		#test with y_test si les resultat sont bons
	except ValueError as e:
		print(e)
	except FileNotFoundError:
		print('Failed to read the dataset')
	


if __name__ == "__main__":
	main()