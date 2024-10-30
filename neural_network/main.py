from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
from Classes.NeuralNetwork import NeuralNetwork
from Classes.NewNeuralNetwork import NewNeuralNetwork
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neural_network.parsing import input_parsing, save_figure, visualization, save_best_performance

from matplotlib.widgets import Button

import os
from Classes.data import Data





def main():

    data = Data()
    best_accuracy = 0
    best_performance = None
    if not os.path.exists("Result"):
        os.makedirs("Result")
    while True:
        # Demande à l'utilisateur de saisir les hyperparamètres
        try:
            result = input_parsing()
            if result is None:
                print("L'utilisateur a quitté l'application.")
                break
            else:
                layers, epochs, learning_rate, use_mini_batch, batch_size = result
            # layers, epochs, learning_rate, use_mini_batch, batch_size = input_parsing()
            # if not layers or not epochs or not learning_rate or not use_mini_batch or not batch_size:
                # break
        except ValueError:
            print("Veuillez entrer des valeurs valides.")
            continue
        #start the creation of the neural
        # neuronne = NeuralNetwork(layers=[data.n_x] + layers + [1])#pour utiliser avec SIgmoid en sortie
        neuronne = NewNeuralNetwork(layers=[data.n_x] + layers + [2])#pour utiliser avec Softtmax en sortie
        if use_mini_batch:
            neuronne.train_mini_batch(data.x_train, data.y_train, epochs, learning_rate, batch_size, data.x_test, data.y_test)
        else:
            # print(data.y_train, data.y_test)
            neuronne.train(data.x_train, data.y_train, epochs, learning_rate, data.x_test, data.y_test)

        # Prédictions sur l'ensemble de test
        print("DATA TEST",data.x_test)
        y_pred_test = neuronne.predict(data.x_test)

        # Prédictions sur l'ensemble d'entraînement (optionnel)


        #start the training
        print("Prediction i did\n", y_pred_test)
        print(y_pred_test.shape, data.y_test.shape)
        data.y_test = data.y_test.T
        print(y_pred_test.shape, data.y_test.shape)
        print("DAtA AT HE END \n", data.y_test )
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

if __name__ == "__main__":
    main()