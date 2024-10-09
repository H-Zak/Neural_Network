import numpy as np
import pandas as pd


class DataInit():
	def __init__(self, path_dataset_file):
		self.data = pd.read_csv(path_dataset_file, header=None)

		self.x = self.data.drop(self.data.columns[1], axis=1)
		self.y = self.data.iloc[:, 1]
		# Splitting data manually
		x_train = self.data.iloc[:,2:4].T
		y_train = self.data.iloc[:, 1].map({'M' : 1, 'B' : 0}).to_numpy()

		# TODO
		self.y_train = data_train[1].reshape(1, -1)
