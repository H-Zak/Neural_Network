import sys
import os
import argparse
import numpy as np
import pandas as pd
import joblib

from Classes.MultilayerPerceptron import MultilayerPerceptron
from Classes.data import Data
from neural_network.parsing import input_parsing, save_figure, visualization, save_best_performance


def run_training():
    print("--- Mode Entraînement ---")
    data = Data()
    if not hasattr(data, 'x_train_scaled_T'):
        print("Erreur lors du chargement des données.")
        return

    best_accuracy = 0
    os.makedirs("Result", exist_ok=True)

    while True:
        try:
            result = input_parsing()
            if result is None:
                break
            layers_hidden, epochs, learning_rate, use_mini_batch, batch_size, patience = result

            if use_mini_batch and batch_size > data.n_train_samples:
                print(f"Erreur : batch ({batch_size}) > échantillons ({data.n_train_samples}).")
                continue

            full_layers = [data.n_x] + layers_hidden + [2]
            print(f"Architecture : {full_layers}")
            neuronne = MultilayerPerceptron(layers=full_layers)

            if use_mini_batch:
                neuronne.train_mini_batch(
                    data.x_train_scaled_T, data.y_train, epochs,
                    learning_rate, batch_size, data.x_val_scaled_T, data.y_val,
                    patience=patience)
            else:
                neuronne.train(
                    data.x_train_scaled_T, data.y_train, epochs,
                    learning_rate, data.x_val_scaled_T, data.y_val,
                    patience=patience)

            y_pred = neuronne.predict(data.x_val_scaled_T)
            true_labels = np.argmax(data.y_val, axis=1)
            pred_labels = np.argmax(y_pred, axis=0)
            accuracy_val = np.mean(pred_labels == true_labels) * 100
            print(f"Précision validation : {accuracy_val:.2f}%")

            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val
                folder_name = (f"epochs_{epochs}_lr_{learning_rate}"
                               f"_batch_{batch_size if use_mini_batch else 0}"
                               f"_accuracy_{round(accuracy_val, 2)}")
                folder_path = os.path.join("Result", folder_name)
                os.makedirs(folder_path, exist_ok=True)

                save_figure(neuronne, os.path.join(folder_path, "cost.png"), 'cost')
                save_figure(neuronne, os.path.join(folder_path, "accuracy.png"), 'accuracy')
                save_best_performance({
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size if use_mini_batch else 0,
                    "layers": neuronne.layers,
                    "accuracy": accuracy_val,
                    "costs": round(neuronne.costs[-1], 6) if neuronne.costs else None,
                    "validation_cost": round(neuronne.validation_cost[-1], 6) if neuronne.validation_cost else None
                }, folder_path)
                print("Nouvelles meilleures performances enregistrées !")

            neuronne.save_weights(os.path.join("Result", "last_model_weights.npz"))

            if sys.platform != 'darwin':
                try:
                    visualization(neuronne)
                except Exception:
                    print("Graphiques sauvegardés dans Result/.")
            else:
                print("Graphiques sauvegardés dans Result/.")

            if input("Relancer un entraînement ? (o/n) : ").lower() != 'o':
                break

        except ValueError as ve:
            print(f"Erreur : {ve}")
        except Exception as e:
            print(f"Erreur inattendue : {e}")
            import traceback
            traceback.print_exc()
            break


def run_prediction(weights_path, input_path):
    print("--- Mode Prédiction ---")

    try:
        model_data = np.load(weights_path, allow_pickle=True)
        layers_config = model_data['layers'].tolist()
        print(f"Architecture : {layers_config}")
    except FileNotFoundError:
        print(f"Erreur : '{weights_path}' non trouvé.")
        return

    try:
        network = MultilayerPerceptron(layers=layers_config)
        network.load_weights(weights_path)
    except Exception as e:
        print(f"Erreur chargement réseau : {e}")
        return

    try:
        scaler = joblib.load("Result/scaler.joblib")
    except FileNotFoundError:
        print("Erreur : 'Result/scaler.joblib' non trouvé. Entraînez un modèle d'abord.")
        return

    try:
        input_df = pd.read_csv(input_path, header=None)
        n_features = network.layers[0]
        n_cols = input_df.shape[1]

        if n_cols == n_features:
            input_data_raw = input_df
        elif n_cols == n_features + 2:
            input_data_raw = input_df.iloc[:, 2:]
        elif n_cols == n_features + 1:
            input_data_raw = input_df.iloc[:, 1:]
        else:
            print(f"Erreur : {n_cols} colonnes, {n_features} features attendues.")
            return

        scaled_data = scaler.transform(input_data_raw)
    except FileNotFoundError:
        print(f"Erreur : '{input_path}' non trouvé.")
        return

    predictions = network.predict(scaled_data.T)
    pred_indices = np.argmax(predictions, axis=0)
    labels = {0: 'B', 1: 'M'}

    print("\n--- Prédictions ---")
    results = pd.DataFrame({'Prediction': [labels[i] for i in pred_indices]})
    print(results)


def main():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron - Train / Predict")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--weights', type=str, default="Result/last_model_weights.npz")
    parser.add_argument('--input', type=str)

    args = parser.parse_args()

    if args.mode == 'train':
        run_training()
    elif args.mode == 'predict':
        if not args.input:
            parser.error("--input est requis pour le mode 'predict'.")
        run_prediction(args.weights, args.input)


if __name__ == "__main__":
    main()
