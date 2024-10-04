from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
from Classes.NeuralNetwork import NeuralNetwork
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from matplotlib.widgets import Button

import matplotlib.pyplot as plt
from Classes.data import Data
import os
import json

# permettre de tapper les hyperparametre dans le terminal
#faire une boucle infini et enregistrer les meilleurs performance (le poids, les hyperparametres)

# Fonction pour enregistrer les meilleures performances
def save_best_performance(best_performance, folder_path):
    performance_file = os.path.join("Result/", "best_performances.json")

    # Lire le fichier s'il existe déjà
    performances = []
    if os.path.exists(performance_file):
        with open(performance_file, "r") as f:
            try:
                performances = json.load(f)
            except json.JSONDecodeError:
                performances = []

    # Ajouter la nouvelle performance
    performances.append(best_performance)

    # Trier les performances par accuracy, la plus élevée en premier
    performances.sort(key=lambda x: x['accuracy'], reverse=True)

    # Sauvegarder le fichier avec les performances triées
    with open(performance_file, "w") as f:
        json.dump(performances, f, indent=4)

def save_figure(neuronne, figure_name, graph_type='cost'):
    plt.figure()
    if graph_type == 'cost':
        plt.plot(neuronne.costs, label='Cost function', color='blue')
        plt.plot(neuronne.validation_cost, label='Validation Cost function', color='orange')
        plt.xlabel('Époques')
        plt.ylabel('Coût')
        plt.title('Évolution du coût pendant l\'apprentissage')
    elif graph_type == 'accuracy':
        plt.plot(neuronne.pred_train, label='Train accuracy', color='blue')
        plt.plot(neuronne.pred_test, label='Test accuracy', color='orange')
        plt.xlabel('Échantillons')
        plt.ylabel('Précision')
        plt.title('Évolution de la précision pendant l\'apprentissage')
    plt.legend()
    plt.savefig(figure_name)
    plt.close()


def validate_positive_int(input_str, label):
    try:
        value = int(input_str)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Erreur : {label} doit être un nombre entier positif.")
        return None

def validate_positive_float(input_str, label):
    try:
        value = float(input_str)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Erreur : {label} doit être un nombre flottant positif.")
        return None
def validate_non_negative_int(input_str, label):
    try:
        value = int(input_str)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Erreur : {label} doit être un nombre entier non-négatif.")
        return None
    
def check_exit(user_input):
    if user_input.lower() == "exit":
        print("Fin du programme.")
        exit()
def validate_layers(input_str):
    if input_str.strip() == "":
        return []  # Pas de couches cachées ajoutées
    try:
        layers = [int(layer) for layer in input_str.split(",")]
        if any(layer <= 0 for layer in layers):
            print("Erreur : Toutes les couches doivent être des nombres entiers positifs.")
            return None
        return layers
    except ValueError:
        print("Erreur : Veuillez entrer une liste de nombres séparés par des virgules (par exemple : 64,32,16).")
        return None
def input_parsing():
    while True:
        user_input = input("Nombre d'époques ou 'exit' pour quitter : ")
        if user_input.lower() == 'exit':
            return
        epochs = validate_positive_int(user_input, "Nombre d'époques")
        if epochs is not None:
            break

    # Taux d'apprentissage
    while True:
        user_input = input("Taux d'apprentissage ou 'exit' pour quitter : ")
        if user_input.lower() == 'exit':
            return
        learning_rate = validate_positive_float(user_input, "Taux d'apprentissage")
        if learning_rate is not None:
            learning_rate = float(user_input)  # Convertir en float si l'utilisateur a entré un nombre valide
            break

    # Taille du mini-batch
    while True:
        user_input = input("Taille du mini-batch (0 pour ne pas utiliser mini-batch) ou 'exit' pour quitter : ")
        if user_input.lower() == 'exit':
            return
        batch_size = validate_non_negative_int(user_input, "Taille du mini-batch")
        if batch_size is not None:
            break

    # Choix des couches
    while True:
        user_input = input("Entrez les tailles des couches séparées par des virgules (par exemple : 64,32,16) ou 'exit' pour quitter : ")
        if user_input.lower() == 'exit':
            return
        layers = validate_layers(user_input)
        if layers is not None:
            break
    use_mini_batch = batch_size > 0
    return layers, epochs, learning_rate, use_mini_batch, batch_size
      
def main():

    data = Data()
    best_accuracy = 0
    best_performance = None
    if not os.path.exists("Result"):
        os.makedirs("Result")
    while True:
        # Demande à l'utilisateur de saisir les hyperparamètres
        try:
            layers, epochs, learning_rate, use_mini_batch, batch_size = input_parsing()
        except ValueError:
            print("Veuillez entrer des valeurs valides.")
            continue
        #start the creation of the neural
        neuronne = NeuralNetwork(layers=[data.n_x] + layers + [1])
        if use_mini_batch:
            neuronne.train_mini_batch(data.x_train, data.y_train, epochs, learning_rate, batch_size, data.x_test, data.y_test)
        else:
            neuronne.train(data.x_train, data.y_train, epochs, learning_rate, data.x_test, data.y_test)

        # Prédictions sur l'ensemble de test
        y_pred_test = neuronne.predict(data.x_test)

        # Prédictions sur l'ensemble d'entraînement (optionnel)


        #start the training 
        accuracy_test = np.mean(y_pred_test == data.y_test) * 100
        print(f"Précision sur l'ensemble de test : {accuracy_test}%")

        if accuracy_test > best_accuracy:
                    best_accuracy = accuracy_test
                    last_cost = round(neuronne.costs[-1], 6) if neuronne.costs else None
                    last_validation_cost = round(neuronne.validation_cost[-1], 6) if neuronne.validation_cost else None
                    folder_name = f"epochs_{epochs}_lr_{learning_rate}_batch_{batch_size}_accuracy_{round(accuracy_test, 2)}"
                    folder_path = os.path.join("Result", folder_name)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    save_figure(neuronne, os.path.join(folder_path, "cost.png"), graph_type='cost')
                    save_figure(neuronne, os.path.join(folder_path, "accuracy.png"), graph_type='accuracy')
                    best_performance = {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "layers": neuronne.layers, 
                        "accuracy": accuracy_test,
                        "costs": last_cost,
                        "validation_cost": last_validation_cost
                    }
                    save_best_performance(best_performance, folder_path)
                    print("Nouvelles meilleures performances enregistrées !")

        # Affichage des résultats avec des commandes simples
        visualization(neuronne)
    

def visualization(neuronne):
    root = tk.Tk()
    root.title("Navigation entre figures")

    global fig, canvas
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Création des boutons
    btn_figure1 = tk.Button(root, text="Cost", command=lambda: plot_figure1(neuronne))
    btn_figure1.pack(side=tk.LEFT)

    btn_figure2 = tk.Button(root, text="Accuracy", command=lambda: plot_figure2(neuronne))
    btn_figure2.pack(side=tk.RIGHT)
    plot_figure1(neuronne)

    root.mainloop()

def plot_figure1(neuronne):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(neuronne.costs, label='Cost function', color='blue')
    ax.plot(neuronne.validation_cost, label='Validation Cost function', color='orange')
    ax.set_xlabel('Époques (par centaine)')
    ax.set_ylabel('Coût')
    ax.set_title('Évolution du coût pendant l\'apprentissage')
    ax.legend()
    canvas.draw()

def plot_figure2(neuronne):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(neuronne.pred_train, label='Train acc', color='blue')
    ax.plot(neuronne.pred_test, label='Test acc', color='orange')
    ax.set_xlabel('Échantillons')
    ax.set_ylabel('Valeur')
    ax.set_title('Comparaison des valeurs prédites et vraies')
    ax.legend()
    canvas.draw()
    


    


if __name__ == "__main__":
    main()