from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
from Classes.MultilayerPerceptron import MultilayerPerceptron
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse # <-- Ajouter
import joblib # <-- Ajouter
from neural_network.parsing import input_parsing, save_figure, visualization, save_best_performance

from matplotlib.widgets import Button

import os
from Classes.data import Data


def run_training():
    """Contient la logique d'entraînement existante."""
    print("--- Mode Entraînement ---")
    data = Data()
    # Vérifier si les données ont été chargées correctement
    if not hasattr(data, 'x_train_scaled_T'):
        print("Erreur lors du chargement des données d'entraînement.")
        return

    best_accuracy = 0
    best_performance = None
    if not os.path.exists("Result"):
        os.makedirs("Result")

    while True:
        try:
            result = input_parsing() # Récupère les hyperparamètres
            if result is None:
                print("L'utilisateur a quitté l'application.")
                break
            else:
                layers_hidden, epochs, learning_rate, use_mini_batch, batch_size = result

            # Vérification taille batch
            if use_mini_batch and batch_size > data.n_train_samples:
                print(f"Erreur : Taille mini-batch ({batch_size}) > nb échantillons train ({data.n_train_samples}).")
                continue

            # Architecture complète (entrée + cachées + sortie)
            # Note: data.n_x = nb features, 2 = nb classes sortie
            full_layers_config = [data.n_x] + layers_hidden + [2]
            print(f"Architecture du réseau : {full_layers_config}")
            neuronne = MultilayerPerceptron(layers=full_layers_config)

            # Entraînement
            if use_mini_batch:
                print(f"Lancement entraînement mini-batch (taille={batch_size})...")
                neuronne.train_mini_batch(data.x_train_scaled_T, data.y_train, epochs, learning_rate, batch_size, data.x_val_scaled_T, data.y_val)
            else:
                print("Lancement entraînement...")
                neuronne.train(data.x_train_scaled_T, data.y_train, epochs, learning_rate, data.x_val_scaled_T, data.y_val)

            # Évaluation sur Validation
            print("Calcul prédictions validation...")
            y_pred_val_one_hot = neuronne.predict(data.x_val_scaled_T)
            true_labels_val = np.argmax(data.y_val, axis=1) # y_val est (N, 2)
            pred_labels_val = np.argmax(y_pred_val_one_hot, axis=0) # predict retourne (2, N)
            accuracy_val = np.mean(pred_labels_val == true_labels_val) * 100
            print(f"Précision sur l'ensemble de validation : {accuracy_val:.2f}%")

            # Sauvegarde meilleure performance
            if accuracy_val > best_accuracy:
                 best_accuracy = accuracy_val
                 last_cost = round(neuronne.costs[-1], 6) if neuronne.costs else None
                 last_validation_cost = round(neuronne.validation_cost[-1], 6) if neuronne.validation_cost else None
                 folder_name = f"epochs_{epochs}_lr_{learning_rate}_batch_{batch_size if use_mini_batch else 0}_accuracy_{round(accuracy_val, 2)}"
                 folder_path = os.path.join("Result", folder_name)
                 if not os.path.exists(folder_path):
                     os.makedirs(folder_path)
                 save_figure(neuronne, os.path.join(folder_path, "cost.png"), graph_type='cost')
                 save_figure(neuronne, os.path.join(folder_path, "accuracy.png"), graph_type='accuracy')
                 best_performance = {
                     "epochs": epochs, "learning_rate": learning_rate,
                     "batch_size": batch_size if use_mini_batch else 0,
                     "layers": neuronne.layers, "accuracy": accuracy_val,
                     "costs": last_cost, "validation_cost": last_validation_cost
                 }
                 save_best_performance(best_performance, folder_path)
                 print("Nouvelles meilleures performances enregistrées !")

            # Sauvegarde poids du dernier modèle
            model_weights_path = os.path.join("Result", "last_model_weights.npz")
            neuronne.save_weights(model_weights_path)

            # Visualisation (conditionnelle)
            if sys.platform != 'darwin':
                 print("\nTentative d'affichage de la visualisation interactive...")
                 try:
                     visualization(neuronne)
                 except Exception as e:
                     print(f"Erreur lors de l'affichage : {e}")
                     print("Graphiques sauvegardés dans le dossier Result.")
            else:
                 print("\nVisualisation interactive sautée sur macOS.")
                 print("Graphiques sauvegardés dans le dossier Result.")

            # Proposer de refaire un entraînement ou quitter ?
            continuer = input("Lancer un autre entraînement ? (o/n) : ").lower()
            if continuer != 'o':
                break

        except ValueError as ve:
             print(f"Erreur de valeur: {ve}")
             print("Veuillez vérifier les paramètres.")
        except Exception as e:
            print(f"Une erreur inattendue est survenue : {e}")
            import traceback
            traceback.print_exc()
            break # Sortir en cas d'erreur imprévue


def run_prediction(weights_path, input_path):
    """Charge un modèle et prédit sur de nouvelles données."""
    print("--- Mode Prédiction ---")
    print(f"Chargement des poids depuis : {weights_path}")
    print(f"Chargement des données depuis : {input_path}")
    print(f"Chargement du scaler depuis : Result/scaler.joblib")

    # --- 1. Charger l'architecture et les poids ---
    try:
        model_data = np.load(weights_path, allow_pickle=True)
        layers_config = model_data['layers'].tolist() # Convertir en liste
        print(f"Architecture chargée : {layers_config}")
    except FileNotFoundError:
        print(f"Erreur : Fichier de poids '{weights_path}' non trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors du chargement des poids : {e}")
        return

    # --- 2. Créer le réseau et charger les poids ---
    try:
        network = MultilayerPerceptron(layers=layers_config)
        network.load_weights(weights_path) # load_weights gère les erreurs internes
    except Exception as e:
        print(f"Erreur lors de l'instanciation ou chargement du réseau : {e}")
        return # Pas la peine de continuer si le réseau n'est pas prêt

    # --- 3. Charger le scaler ---
    try:
        scaler = joblib.load("Result/scaler.joblib")
    except FileNotFoundError:
        print("Erreur : Fichier scaler 'Result/scaler.joblib' non trouvé. Assurez-vous d'avoir entraîné un modèle d'abord.")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du scaler : {e}")
        return

    # --- 4. Charger et Prétraiter les données d'entrée ---
    try:
        # Supposons que le CSV d'entrée contient les features aux mêmes positions
        # que data_cancer.csv (colonnes 2 à la fin), sans ID ni label, et sans en-tête.
        # S'il y a un ID ou des labels, il faut adapter le chargement.
        input_df = pd.read_csv(input_path, header=None)
        # Extraire les features (supposons toutes les colonnes si pas d'ID/label)
        # Si l'input a les mêmes 32 colonnes que data_cancer.csv mais sans labels,
        # il faudrait peut-être ne prendre que les colonnes 2 à 31 ?
        # Soyons flexible : prenons toutes les colonnes lues.
        input_data_raw = input_df.iloc[:, :]

        # Vérifier si le nombre de features correspond
        if input_data_raw.shape[1] != network.layers[0]: # network.layers[0] est n_x
             print(f"Erreur: Le nombre de features dans {input_path} ({input_data_raw.shape[1]})"
                   f" ne correspond pas à la couche d'entrée du modèle ({network.layers[0]}).")
             return

        # Mise à l'échelle
        scaled_input_data = scaler.transform(input_data_raw)
        # Transposition
        scaled_input_data_T = scaled_input_data.T # Shape (n_features, n_samples)

    except FileNotFoundError:
        print(f"Erreur : Fichier d'entrée '{input_path}' non trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors du chargement ou prétraitement des données d'entrée : {e}")
        return

    # --- 5. Faire la Prédiction ---
    print("Prédiction en cours...")
    try:
        predictions_one_hot = network.predict(scaled_input_data_T) # Shape (2, N)
        pred_indices = np.argmax(predictions_one_hot, axis=0) # Indices (0 ou 1)
        # Mapper les indices aux labels
        label_map = {0: 'B', 1: 'M'}
        predictions_labels = [label_map[i] for i in pred_indices]
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return

    # --- 6. Afficher les Résultats ---
    print("\n--- Prédictions ---")
    if input_df.shape[0] == len(predictions_labels):
        # Créer un DataFrame pour un affichage clair
        results_df = pd.DataFrame({'Prediction': predictions_labels})
        # Si le fichier d'input avait une colonne ID (ex: colonne 0), on pourrait l'ajouter
        # input_ids = input_df.iloc[:, 0]
        # results_df.insert(0, 'ID', input_ids)
        print(results_df)
        # Optionnel: Sauvegarder les prédictions dans un fichier
        # results_df.to_csv("predictions.csv", index=False)
    else:
        # Affichage simple si problème de taille
        print(predictions_labels)

    # Optionnel : Évaluation si les vrais labels sont disponibles
    # true_labels_path = ...
    # evaluate(predictions_labels, true_labels_path) # Fonction à créer


def main():
    parser = argparse.ArgumentParser(description="Entraînement ou Prédiction avec un Réseau de Neurones Multicouches.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help="Mode d'exécution : 'train' pour entraîner un modèle, 'predict' pour faire des prédictions.")
    parser.add_argument('--weights', type=str, default="Result/last_model_weights.npz",
                        help="Chemin vers le fichier de poids (.npz) à charger (pour predict) ou sauvegarder (par défaut pour train).")
    parser.add_argument('--input', type=str,
                        help="Chemin vers le fichier CSV de données d'entrée (requis pour predict).")

    args = parser.parse_args()

    if args.mode == 'train':
        run_training() # Lancer la logique d'entraînement
    elif args.mode == 'predict':
        if not args.input:
            parser.error("--input est requis pour le mode 'predict'.")
        run_prediction(args.weights, args.input) # Lancer la logique de prédiction




if __name__ == "__main__":
    main()
