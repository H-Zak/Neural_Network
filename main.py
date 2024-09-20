from sklearn.model_selection import train_test_split
import pandas as pd

def main():
	print("Start")
	data =  pd.read_csv("./data_cancer.csv")
	x_train, x_tet, y_train, y_test = train_test_split(x,y, test_size=0.2,random=42) 
	


if __name__ == "__main__":
	main()