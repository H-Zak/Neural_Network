import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib # <-- Ajouter
import os     # <-- Ajouter

class Data():
    def __init__(self):
        # Créer le dossier Result s'il n'existe pas (sécurité)
        os.makedirs("Result", exist_ok=True) # <-- Ajouter

        # Charger data_cancer.csv
        try:
            self.data =  pd.read_csv("./dataset/data_cancer.csv", header=None)
        except FileNotFoundError:
            print("Erreur : Le fichier ./dataset/data_cancer.csv est introuvable.")
            # Vous pourriez vouloir quitter ou lever une exception ici
            return

        self.x = self.data.iloc[:, 2:] # Features
        self.y = self.data.iloc[:, 1]  # Labels 'M'/'B'
        self.y = self.y.map({'B': 0, 'M': 1})
        print("Labels (0=B, 1=M):\n", self.y.head()) # Afficher quelques labels mappés
        self.y_one_hot = pd.get_dummies(self.y).astype(int)
        # S'assurer que les colonnes one-hot sont bien 0 et 1
        if list(self.y_one_hot.columns) != [0, 1]:
             # Si les colonnes sont nommées différemment, les renommer peut être nécessaire
             # Ou s'assurer que l'ordre est cohérent. Par défaut, get_dummies trie.
            #  print("Colonnes One-Hot:", self.y_one_hot.columns)
             # Assumons que la première colonne est la classe 0 (B), la seconde est 1 (M)
             self.y_one_hot.columns = [0, 1] # Forcer les noms si nécessaire

        # print("Labels One-Hot (B=[1 0], M=[0 1]):\n", self.y_one_hot.head())

        # Split Train/Validation (80/20)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y_one_hot, test_size=0.2, random_state=42, stratify=self.y # stratify est bien pour la classification
        )

        scaler = StandardScaler()
        # Adapter (fit) sur l'entraînement ET transformer l'entraînement
        self.x_train_scaled = scaler.fit_transform(self.x_train)
        # Transformer SEULEMENT la validation avec le scaler adapté
        self.x_val_scaled = scaler.transform(self.x_val)

        # ====> SAUVEGARDER LE SCALER <====
        scaler_filepath = "Result/scaler.joblib"
        try:
            joblib.dump(scaler, scaler_filepath)
            print(f"Scaler sauvegardé dans {scaler_filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du scaler : {e}")
        # ================================

        # Transposer pour le réseau (features x samples)
        self.x_train_scaled_T = self.x_train_scaled.T
        self.x_val_scaled_T = self.x_val_scaled.T

        # Garder y_train et y_val en format (samples, features/classes) pour le moment
        # La transposition sera gérée dans la classe réseau si besoin
        self.y_train = self.y_train.to_numpy()
        self.y_val = self.y_val.to_numpy()

        self.n_x = self.x_train_scaled_T.shape[0] # Nombre de features
        self.n_train_samples = self.x_train_scaled_T.shape[1] # Nombre d'échantillons train
        self.n_val_samples = self.x_val_scaled_T.shape[1]     # Nombre d'échantillons val

        # print(f"Données d'entraînement : X({self.x_train_scaled_T.shape}), Y({self.y_train.shape})")
        # print(f"Données de validation : X({self.x_val_scaled_T.shape}), Y({self.y_val.shape})")
