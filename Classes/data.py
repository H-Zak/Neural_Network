import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class Data():
    def __init__(self):
        os.makedirs("Result", exist_ok=True)

        self.data = pd.read_csv("./dataset/data_cancer.csv", header=None)

        self.x = self.data.iloc[:, 2:]
        self.y = self.data.iloc[:, 1].map({'B': 0, 'M': 1})

        self.y_one_hot = pd.get_dummies(self.y).astype(int)
        if list(self.y_one_hot.columns) != [0, 1]:
            self.y_one_hot.columns = [0, 1]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y_one_hot,
            test_size=0.2, random_state=42, stratify=self.y
        )

        scaler = StandardScaler()
        self.x_train_scaled = scaler.fit_transform(self.x_train)
        self.x_val_scaled = scaler.transform(self.x_val)

        joblib.dump(scaler, "Result/scaler.joblib")

        self.x_train_scaled_T = self.x_train_scaled.T
        self.x_val_scaled_T = self.x_val_scaled.T

        self.y_train = self.y_train.to_numpy()
        self.y_val = self.y_val.to_numpy()

        self.n_x = self.x_train_scaled_T.shape[0]
        self.n_train_samples = self.x_train_scaled_T.shape[1]
        self.n_val_samples = self.x_val_scaled_T.shape[1]
