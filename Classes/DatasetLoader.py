import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DatasetLoader():
	def __init__(self, path_dataset_file : str):

        

		self.data : pd.DataFrame = pd.read_csv(path_dataset_file, header=None)

		self.x = self.data.drop([self.data.columns[0], self.data.columns[1]], axis=1)
		self.y = self.data.iloc[:, 1]

		# # Splitting data manually
		# x_train = self.data.iloc[:,2:4].T
		# y_train = self.data.iloc[:, 1]
		# # self.y_train = data_train[1].reshape(1, -1)
		
		x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
		scaler = StandardScaler()

		self.x_train = scaler.fit_transform(x_train.T) 
		self.x_test = scaler.fit_transform(x_test.T)
		# self.y_train = y_train.map({'M': 1, 'B': 0}).to_numpy().flatten()
		# self.y_test = y_test.map({'M': 1, 'B': 0}).to_numpy().flatten()

		self.y_train = y_train.map({'M': 1, 'B': 0}).to_numpy().reshape(1, -1)
		self.y_test = y_test.map({'M': 1, 'B': 0}).to_numpy().reshape(1, -1)
		
		self.y_train_hot_encoding = self.one_hot_encoding(self.y_train, num_classes=2)
		# print("Input layer")
		# print(self.x_train.shape)
		# print("y train:")
		# print(self.y_train_hot_encoding.shape)

	def one_hot_encoding(self, y, num_classes):
		"""
		Converts a 1D array of labels to one-hot encoded format.
		Args:
		y (np.ndarray): Array of labels (shape: [1, batch_size]).
		num_classes (int): Number of output classes (for binary classification, this is 2).
		
		Returns:
		np.ndarray: One-hot encoded labels (shape: [num_classes, batch_size]).
		"""
		return np.eye(num_classes)[y.reshape(-1)].T

	def get_input_shape(self) -> int:
		return self.x_train.shape[0]

	def get_train_data(self) -> tuple:
		return (self.x_train, self.y_train)

	def get_test_data(self) -> tuple:
		return (self.x_test, self.y_test)