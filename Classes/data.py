import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Data():
	def __init__(self):
		self.data =  pd.read_csv("./dataset/data_cancer.csv")
		self.x = self.data.drop(self.data.columns[1], axis=1)
		self.y = self.data.iloc[:, 1]
		self.y = self.y.map({'B': 0, 'M': 1})
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=0.2,random_state=42)


		scaler = StandardScaler()
		self.x_train = scaler.fit_transform(self.x_train)
		self.x_test = scaler.transform(self.x_test)
		self.x_train = self.x_train.T  # Si x_train est de dimension (m, n_x)
		self.x_test = self.x_test.T
		self.y_train = self.y_train.to_numpy().reshape(1, -1)  # Convertir en tableau NumPy et redimensionner
		self.y_test = self.y_test.to_numpy().reshape(1, -1)

		self.n_x = self.x_train.shape[0]

		self.n = int(self.x_train.shape[1])

