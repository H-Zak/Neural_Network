from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
from Classes.NeuralNetwork import NeuralNetwork
import numpy as np

import matplotlib.pyplot as plt

def main():
	print("Start")

	data =  pd.read_csv("./dataset/data_cancer.csv")
	x = data.drop(data.columns[1], axis=1)
	y = data.iloc[:, 1]
	y = y.map({'B': 0, 'M': 1})
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)


	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	x_train = x_train.T  # Si x_train est de dimension (m, n_x)
	x_test = x_test.T
	y_train = y_train.to_numpy().reshape(1, -1)  # Convertir en tableau NumPy et redimensionner
	y_test = y_test.to_numpy().reshape(1, -1)

	print(f"x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}")
	print(f"x_test.shape = {x_test.shape}, y_test.shape = {y_test.shape}")


	n_x = x_train.shape[0]




	#start the creation of the neural
	neuronne = NeuralNetwork(layers=[n_x,20,20,1])
	neuronne.train_mini_batch(x_train,y_train, 20000, 0.01, 128)
	# neuronne.train(x_train,y_train, 20000, 0.01)

	# Prédictions sur l'ensemble de test
	y_pred_test = neuronne.predict(x_test)

	# Prédictions sur l'ensemble d'entraînement (optionnel)
	y_pred_train = neuronne.predict(x_train)

	#start the training 
	accuracy_test = np.mean(y_pred_test == y_test) * 100
	print(f"Précision sur l'ensemble de test : {accuracy_test}%")

	#test with y_test si les resultat sont bons
	plt.plot(neuronne.costs)
	plt.xlabel('Époques (par centaine)')
	plt.ylabel('Coût')
	plt.title('Évolution du coût pendant l\'apprentissage')
	plt.show()
	


if __name__ == "__main__":
	main()