import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def save_best_performance(best_performance, folder_path):
    performance_file = os.path.join("Result", "best_performances.json")

    performances = []
    if os.path.exists(performance_file):
        with open(performance_file, "r") as f:
            try:
                performances = json.load(f)
            except json.JSONDecodeError:
                performances = []

    performances.append(best_performance)
    performances.sort(key=lambda x: x['accuracy'], reverse=True)

    with open(performance_file, "w") as f:
        json.dump(performances, f, indent=4)


def save_figure(neuronne, figure_name, graph_type='cost'):
    plt.figure()
    if graph_type == 'cost':
        plt.plot(neuronne.costs, label='Training cost', color='blue')
        plt.plot(neuronne.validation_cost, label='Validation cost', color='orange')
        plt.xlabel('Époques (x100)')
        plt.ylabel('Coût')
        plt.title('Évolution du coût')
    elif graph_type == 'accuracy':
        plt.plot(neuronne.pred_train, label='Training accuracy', color='blue')
        plt.plot(neuronne.pred_val, label='Validation accuracy', color='orange')
        plt.xlabel('Époques (x100)')
        plt.ylabel('Précision (%)')
        plt.title('Évolution de la précision')
    plt.legend()
    os.makedirs(os.path.dirname(figure_name), exist_ok=True)
    plt.savefig(figure_name)
    plt.close()


def validate_positive_int(input_str, label):
    try:
        value = int(input_str)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Erreur : {label} doit être un entier positif.")
        return None


def validate_positive_float(input_str, label):
    try:
        value = float(input_str)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Erreur : {label} doit être un flottant positif.")
        return None


def validate_non_negative_int(input_str, label):
    try:
        value = int(input_str)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Erreur : {label} doit être un entier >= 0.")
        return None


def validate_layers(input_str):
    try:
        layers = [int(layer) for layer in input_str.split(",")]
        if any(layer <= 0 for layer in layers):
            print("Erreur : toutes les couches doivent être des entiers positifs.")
            return None
        if len(layers) < 2:
            print("Erreur : au moins 2 couches cachées requises (ex: 64,32).")
            return None
        return layers
    except ValueError:
        print("Erreur : format attendu = nombres séparés par des virgules (ex: 64,32,16).")
        return None


def input_parsing():
    while True:
        user_input = input("Nombre d'époques (ou 'exit') : ")
        if user_input.lower() == 'exit':
            return None
        epochs = validate_positive_int(user_input, "Nombre d'époques")
        if epochs is not None:
            break

    while True:
        user_input = input("Taux d'apprentissage (ou 'exit') : ")
        if user_input.lower() == 'exit':
            return None
        learning_rate = validate_positive_float(user_input, "Taux d'apprentissage")
        if learning_rate is not None:
            break

    while True:
        user_input = input("Taille mini-batch (0 = full batch, ou 'exit') : ")
        if user_input.lower() == 'exit':
            return None
        batch_size = validate_non_negative_int(user_input, "Taille mini-batch")
        if batch_size is not None:
            break

    while True:
        user_input = input("Couches cachées séparées par virgules (ex: 64,32) ou 'exit' : ")
        if user_input.lower() == 'exit':
            return None
        layers = validate_layers(user_input)
        if layers is not None:
            break

    while True:
        user_input = input("Early stopping patience (0 = désactivé, ou 'exit') : ")
        if user_input.lower() == 'exit':
            return None
        patience = validate_non_negative_int(user_input, "Patience")
        if patience is not None:
            break

    use_mini_batch = batch_size > 0
    return layers, epochs, learning_rate, use_mini_batch, batch_size, patience


def visualization(neuronne):
    root = tk.Tk()
    root.title("Courbes d'apprentissage")

    fig = plt.Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def plot_cost():
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(neuronne.costs, label='Training cost', color='blue')
        ax.plot(neuronne.validation_cost, label='Validation cost', color='orange')
        ax.set_xlabel('Époques (x100)')
        ax.set_ylabel('Coût')
        ax.set_title('Évolution du coût')
        ax.legend()
        canvas.draw()

    def plot_accuracy():
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(neuronne.pred_train, label='Training accuracy', color='blue')
        ax.plot(neuronne.pred_val, label='Validation accuracy', color='orange')
        ax.set_xlabel('Époques (x100)')
        ax.set_ylabel('Précision (%)')
        ax.set_title('Évolution de la précision')
        ax.legend()
        canvas.draw()

    tk.Button(root, text="Cost", command=plot_cost).pack(side=tk.LEFT)
    tk.Button(root, text="Accuracy", command=plot_accuracy).pack(side=tk.RIGHT)
    plot_cost()

    root.mainloop()
