import sys
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from Classes.MultilayerPerceptron import MultilayerPerceptron
from neural_network.parsing import input_parsing, save_figure, visualization, save_best_performance


def run_split(dataset_path):
    print("--- Mode Split ---")
    data = pd.read_csv(dataset_path, header=None)

    x = data.iloc[:, 2:]
    y = data.iloc[:, 1].map({'B': 0, 'M': 1})

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train))
    x_val_scaled = pd.DataFrame(scaler.transform(x_val))

    os.makedirs("Result", exist_ok=True)
    joblib.dump(scaler, "Result/scaler.joblib")

    train_df = pd.concat([y_train.reset_index(drop=True), x_train_scaled], axis=1)
    val_df = pd.concat([y_val.reset_index(drop=True), x_val_scaled], axis=1)

    train_df.to_csv("dataset/data_training.csv", index=False, header=False)
    val_df.to_csv("dataset/data_validation.csv", index=False, header=False)

    print(f"x_train shape : ({x_train_scaled.shape[0]}, {x_train_scaled.shape[1]})")
    print(f"x_valid shape : ({x_val_scaled.shape[0]}, {x_val_scaled.shape[1]})")
    print(f"Sauvegardé : dataset/data_training.csv, dataset/data_validation.csv")


def parse_labels_column(col):
    if col.dtype == object:
        return col.map({'B': 0, 'M': 1}).values.astype(float)
    return col.values.astype(float)


def run_training(args):
    print("--- Mode Entraînement ---")

    if not os.path.exists("dataset/data_training.csv") or not os.path.exists("dataset/data_validation.csv"):
        print("Fichiers split introuvables. Lancement du split automatique...")
        run_split("dataset/data_cancer.csv")

    train_df = pd.read_csv("dataset/data_training.csv", header=None)
    val_df = pd.read_csv("dataset/data_validation.csv", header=None)

    y_train_labels = train_df.iloc[:, 0].values
    x_train = train_df.iloc[:, 1:].values.T
    y_val_labels = val_df.iloc[:, 0].values
    x_val = val_df.iloc[:, 1:].values.T

    y_train = np.zeros((len(y_train_labels), 2))
    y_train[np.arange(len(y_train_labels)), y_train_labels.astype(int)] = 1
    y_val = np.zeros((len(y_val_labels), 2))
    y_val[np.arange(len(y_val_labels)), y_val_labels.astype(int)] = 1

    n_x = x_train.shape[0]
    n_train = x_train.shape[1]

    print(f"x_train shape : ({n_train}, {n_x})")
    print(f"x_valid shape : ({x_val.shape[1]}, {n_x})")

    best_accuracy = 0
    os.makedirs("Result", exist_ok=True)

    # Si les hyperparamètres sont passés en CLI, un seul run
    # Sinon, boucle interactive
    cli_mode = args.layer is not None and args.epochs is not None

    while True:
        try:
            if cli_mode:
                layers_hidden = args.layer
                epochs = args.epochs
                learning_rate = args.learning_rate
                batch_size = args.batch_size
                patience = args.patience
                use_mini_batch = batch_size > 0
            else:
                result = input_parsing()
                if result is None:
                    break
                layers_hidden, epochs, learning_rate, use_mini_batch, batch_size, patience = result

            if use_mini_batch and batch_size > n_train:
                print(f"Erreur : batch ({batch_size}) > échantillons ({n_train}).")
                if cli_mode:
                    break
                continue

            full_layers = [n_x] + layers_hidden + [2]
            print(f"Architecture : {full_layers}")
            neuronne = MultilayerPerceptron(layers=full_layers)

            if use_mini_batch:
                neuronne.train_mini_batch(
                    x_train, y_train, epochs,
                    learning_rate, batch_size, x_val, y_val,
                    patience=patience)
            else:
                neuronne.train(
                    x_train, y_train, epochs,
                    learning_rate, x_val, y_val,
                    patience=patience)

            y_pred = neuronne.predict(x_val)
            true_labels = np.argmax(y_val, axis=1)
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

            neuronne.save_weights(os.path.join("Result", "last_model_weights.npz"))
            print(f"> saving model 'Result/last_model_weights.npz' to disk...")

            try:
                visualization(neuronne)
            except Exception:
                print("Graphiques sauvegardés dans Result/.")

            if cli_mode:
                break

            if input("Relancer un entraînement ? (o/n) : ").lower() != 'o':
                break

        except ValueError as ve:
            print(f"Erreur : {ve}")
            if cli_mode:
                break
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

        labels_col = None
        if n_cols == n_features:
            input_data_raw = input_df
        elif n_cols == n_features + 1:
            labels_col = parse_labels_column(input_df.iloc[:, 0])
            input_data_raw = input_df.iloc[:, 1:]
        elif n_cols == n_features + 2:
            labels_col = parse_labels_column(input_df.iloc[:, 1])
            input_data_raw = input_df.iloc[:, 2:]
        else:
            print(f"Erreur : {n_cols} colonnes, {n_features} features attendues.")
            return

        scaled_data = scaler.transform(input_data_raw)
    except FileNotFoundError:
        print(f"Erreur : '{input_path}' non trouvé.")
        return

    output, _, _ = network.forward_propagation(scaled_data.T)
    pred_indices = np.argmax(output, axis=0)
    label_map = {0: 'B', 1: 'M'}

    print("\n--- Prédictions ---")
    results = pd.DataFrame({'Prediction': [label_map[i] for i in pred_indices]})
    print(results)

    if labels_col is not None and not np.any(np.isnan(labels_col)):
        p = output[1]
        y = labels_col
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1 - epsilon)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        accuracy = np.mean(pred_indices == y) * 100
        print(f"\n--- Évaluation ---")
        print(f"Binary cross-entropy : {bce:.6f}")
        print(f"Accuracy : {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron - Split / Train / Predict")
    parser.add_argument('--mode', type=str, required=True, choices=['split', 'train', 'predict'])
    parser.add_argument('--dataset', type=str, default="dataset/data_cancer.csv")
    parser.add_argument('--weights', type=str, default="Result/last_model_weights.npz")
    parser.add_argument('--input', type=str)

    # Arguments CLI pour le training (optionnels — sinon interactif)
    parser.add_argument('--layer', type=int, nargs='+', help="Couches cachées (ex: --layer 24 24)")
    parser.add_argument('--epochs', type=int, help="Nombre d'époques")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--patience', type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'split':
        run_split(args.dataset)
    elif args.mode == 'train':
        run_training(args)
    elif args.mode == 'predict':
        if not args.input:
            parser.error("--input est requis pour le mode 'predict'.")
        run_prediction(args.weights, args.input)


if __name__ == "__main__":
    main()
