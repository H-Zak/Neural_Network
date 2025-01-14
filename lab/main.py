from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Classes imports
from Classes.DatasetLoader import DatasetLoader
from Classes.Model import Model
from Classes.NeuralNetwork import NeuralNetwork
# Loss function
# from modules.loss_function import binary_cross_entropy

data = pd.read_csv("./dataset/data_cancer.csv")
m, n = data.shape
  
def init_params():
    W1 = np.random.rand(10, 455) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2



def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    print(W1.shape)
    print(X.shape)
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
    # for i in range(iterations):
    #     Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    #     dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
    #     W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    #     if i % 10 == 0:
    #         print("Iteration: ", i)
    #         predictions = get_predictions(A2)
    #         print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def main():
	try:
		data = pd.read_csv("./dataset/data_cancer.csv")
		m, n = data.shape
		X = data.drop([data.columns[0], data.columns[1]], axis=1)
		Y = data.iloc[:, 1].map({'M': 1, 'B': 0}).to_numpy()

		scaler = StandardScaler()

		alpha = 0.1
		X_test = scaler.fit_transform(X[0:113])
		Y_test = Y[0:113]

		X_train = scaler.fit_transform(X[113:])
		Y_train = Y[113:]
		
		W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, 500)


		# data_loader = DatasetLoader("./dataset/data_cancer.csv")

		# print(y_train.shape)
		# neural_network = NeuralNetwork(input_shape=x_train.shape[0], hidden_layers=[24, 24, 24], output_shape=1)
		# # Initialize model
		# model = Model(network=neural_network, 
        #               data_train=(x_train, y_train),
        #               data_valid=([], []),
        #               loss_function=binary_cross_entropy, 
        #               learning_rate=0.1, 
        #               batch_size=2,
        #               epochs=5000)
		# # Start the training
		# model.train()
	# except ValueError as e:
	# 	print(e)
	except FileNotFoundError:
		print('Failed to read the dataset')

if __name__ == "__main__":
	main()