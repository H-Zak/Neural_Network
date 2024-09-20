from sklearn.model_selection import train_test_split
import pandas as pd

def main():
	print("Start")
	data =  pd.read_csv("./data_cancer.csv")
	x = data.drop(data.columns[1], axis=1)
	y = data.iloc[:, 1]
	x_train, x_tet, y_train, y_test = train_test_split(x,y, test_size=0.2,random=42)
	#start the creation of the neural

	#start the training 

	#test with y_test si les resultat sont bons 
	


if __name__ == "__main__":
	main()