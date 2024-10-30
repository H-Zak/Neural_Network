import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Data():
	def __init__(self):
		self.data =  pd.read_csv("./dataset/data_cancer_1.csv")
		self.x = self.data.drop(self.data.columns[1], axis=1)
		self.y = self.data.iloc[:, 1]
		self.y = self.y.map({'B': 0, 'M': 1})
		print(self.y)
		self.y_one_hot = pd.get_dummies(self.y, prefix='class').astype(int)
		print(self.y_one_hot)
		# print("Self y one shot\n ",self.y_one_hot)
		# self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=0.2,random_state=42)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y_one_hot, test_size=0.5,random_state=42)

		scaler = StandardScaler()
		self.x_train = scaler.fit_transform(self.x_train)
		self.x_test = scaler.transform(self.x_test)
		self.x_train = self.x_train.T  # Si x_train est de dimension (m, n_x)
		self.x_test = self.x_test.T
		# self.y_train = self.y_train.to_numpy().reshape(1, -1)  # Convertir en tableau NumPy et redimensionner
		# self.y_test = self.y_test.to_numpy().reshape(1, -1)
		self.y_train = self.y_train.to_numpy()  # Convertir en tableau NumPy et redimensionner
		self.y_test = self.y_test.to_numpy()
		self.n_x = self.x_train.shape[0]

		print("Data test", self.y_train,self.y_test)
		self.n = int(self.x_train.shape[1])

		# print("\nself x train\n", self.x_train, "\nself.y_train\n", self.y_train, "\nself x_test\n", self.x_test, "\nself y_test\n", self.y_test)
