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

		print(f"All examples shape {self.x.shape}")
		scaler = StandardScaler()

		self.x_train = scaler.fit_transform(x_train.T) 
		self.x_test = scaler.fit_transform(x_test.T)
		# self.y_train = y_train.map({'M': 1, 'B': 0}).to_numpy().flatten()
		# self.y_test = y_test.map({'M': 1, 'B': 0}).to_numpy().flatten()

		self.y_train = y_train.map({'M': 1, 'B': 0}).to_numpy().reshape(1, -1)
		self.y_test = y_test.map({'M': 1, 'B': 0}).to_numpy().reshape(1, -1)
		
		print(self.x_train.shape)
		print(self.y_train.shape)

	def get_input_shape(self) -> int:
		return self.x_train.shape[0]

	def get_train_data(self) -> tuple:
		return (self.x_train, self.y_train)

	def get_test_data(self) -> tuple:
		return (self.x_test, self.y_test)