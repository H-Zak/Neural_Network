# Réseau de Neurones Multicouches pour la Classification du Cancer du Sein

## Introduction

Ce projet consiste en l'implémentation "from scratch" (sans bibliothèques d'IA dédiées) d'un Perceptron Multicouches (Multilayer Perceptron - MLP) en Python. L'objectif est de prédire la nature (maligne ou bénigne) de tumeurs du sein en se basant sur le jeu de données "Wisconsin Breast Cancer". Ce travail sert d'introduction pratique aux concepts fondamentaux des réseaux de neurones artificiels.

*(Basé sur le sujet "Machine Learning project - Multilayer Perceptron - Version: 5.1" [cite: 11, 12])*

## But du Projet et Apprentissages (Partie Pédagogique)

L'objectif principal de ce projet était double :

1.  **Implémenter un MLP de A à Z** : Comprendre et coder les mécanismes internes d'un réseau de neurones, notamment les algorithmes essentiels que sont la **propagation avant (feedforward)**, la **rétropropagation de l'erreur (backpropagation)** et l'optimisation par **descente de gradient**[cite: 36, 50].
2.  **Appliquer le Modèle** : Utiliser le réseau implémenté pour résoudre un problème de classification concret (classification du cancer du sein Wisconsin [cite: 16, 55]) et évaluer sa performance.

Au cours de ce projet, les concepts et compétences suivants ont été explorés et mis en œuvre :

* **Architecture Neuronale**: Conception d'un réseau avec des couches denses (entrée, cachées, sortie).
* **Fonctions d'Activation**: Compréhension et implémentation de Sigmoid, ReLU[cite: 35], et **Softmax** (pour obtenir une sortie probabiliste sur la couche finale [cite: 5, 64]).
* **Propagation Avant**: Calcul de la sortie du réseau pour une entrée donnée.
* **Fonction de Coût**: Utilisation de la Cross-Entropy Catégorielle pour quantifier l'erreur du modèle, adaptée ici pour la classification binaire avec sortie Softmax.
* **Rétropropagation & Descente de Gradient**: Calcul des gradients de l'erreur par rapport aux poids et biais, et mise à jour de ces paramètres pour minimiser le coût.
* **Prétraitement des Données**: Importance de la mise à l'échelle des features (avec `StandardScaler`) pour améliorer l'apprentissage.
* **Gestion du Dataset**: Séparation des données en ensembles d'entraînement et de validation pour une évaluation fiable du modèle[cite: 60].
* **Évaluation du Modèle**: Suivi de l'accuracy et du coût sur les ensembles d'entraînement et de validation ; interprétation des courbes d'apprentissage[cite: 66].
* **Persistance du Modèle**: Sauvegarde des poids entraînés (`.npz`) et du scaler (`.joblib`) pour une utilisation ultérieure en prédiction.
* **Interface Utilisateur**: Utilisation d'`argparse` pour créer une interface en ligne de commande permettant de choisir entre les modes entraînement et prédiction.
* **Tests Unitaires**: Mise en place de tests (`unittest`) pour vérifier le bon fonctionnement des composants individuels du réseau.
* **Bonnes Pratiques Python**: Structuration du projet (dossiers, `__init__.py`), gestion des dépendances (`requirement.txt`, `venv`).
* **Débogage Multi-plateforme**: Identification et contournement de problèmes spécifiques à l'OS (visualisation Tkinter/Matplotlib sur macOS).

## Fonctionnalités Implémentées

* Classe `MultilayerPerceptron` implémentant le réseau de neurones.
* Architecture configurable avec au moins deux couches cachées possibles[cite: 62].
* Fonctions d'activation Sigmoid/ReLU pour les couches cachées et Softmax pour la sortie[cite: 5].
* Entraînement par descente de gradient (option mini-batch implémentée).
* Chargement et prétraitement automatique du dataset Wisconsin (scaling, one-hot encoding).
* Séparation des données en ensembles d'entraînement (80%) et de validation (20%).
* Suivi et sauvegarde de l'historique du coût et de la précision (entraînement et validation).
* Sauvegarde automatique des poids du dernier modèle entraîné (`last_model_weights.npz`).
* Sauvegarde automatique du scaler de données (`scaler.joblib`).
* Sauvegarde des métriques et des graphiques (`.png`) correspondant à la **meilleure performance** de validation obtenue.
* Interface en ligne de commande (`neural_network/main.py`) avec deux modes:
    * `train`: Entraîne un nouveau modèle (hyperparamètres via inputs interactifs).
    * `predict`: Charge un modèle et un scaler sauvegardés pour prédire sur de nouvelles données fournies via un fichier CSV.
* Ensemble de tests unitaires pour valider les fonctions d'activation, de coût, d'initialisation et de sauvegarde/chargement.

## Structure du Projet
<pre> <code>```plaintext My_project/ ├── Classes/ │ ├── __init__.py │ ├── MultilayerPerceptron.py │ └── data.py ├── dataset/ │ ├── data_cancer.csv │ └── ... ├── docs/ │ └── ... ├── neural_network/ │ ├── __init__.py │ ├── main.py │ └── parsing.py ├── Result/ │ ├── best_performances.json │ ├── last_model_weights.npz │ ├── scaler.joblib │ └── epochs_...acc.../ │ ├── cost.png │ └── accuracy.png ├── test/ │ ├── __init__.py │ └── test.py ├── README.md ├── requirement.txt └── setup.py ```</code> </pre>

## Prérequis

* Python 3.x
* Bibliothèques listées dans `requirement.txt` (NumPy, Pandas, Scikit-learn, Matplotlib, Joblib)

## Installation et Configuration

1.  Clonez le dépôt (si applicable).
2.  Naviguez jusqu'au dossier racine du projet (`MY_project/`).
3.  (Recommandé) Créez et activez un environnement virtuel Python :
    ```bash
    python3 -m venv venv_mlp
    source venv_mlp/bin/activate  # Sur Mac/Linux
    # ou .\venv_mlp\Scripts\activate sur Windows
    ```
4.  Installez les dépendances :
    ```bash
    pip install -r requirement.txt
    ```

## Utilisation

Le script principal est `neural_network/main.py`. Exécutez-le **depuis le dossier racine (`My_project/`)** en utilisant l'option `-m`.

**1. Entraînement**

* Lancez la commande :
    ```bash
    python3 -m neural_network.main --mode train
    ```
* Le programme vous demandera interactivement les hyperparamètres (époques, taux d'apprentissage, taille du mini-batch, couches cachées).
* L'entraînement se lancera, affichant le coût périodiquement.
* À la fin, la précision de validation sera affichée.
* Les fichiers `Result/last_model_weights.npz` et `Result/scaler.joblib` seront créés/mis à jour.
* Si la performance est la meilleure, les métriques et graphiques seront sauvegardés dans `Result/` et `best_performances.json` sera mis à jour.

**2. Prédiction**

* **Préparez un fichier CSV d'entrée** (ex: `dataset/predict_input.csv`). Ce fichier doit contenir uniquement les 30 colonnes de features, sans en-tête, et dans le même ordre que les données d'entraînement.
* Lancez la commande :
    ```bash
    python3 -m neural_network.main --mode predict --input CHEMIN/VERS/VOTRE/FICHIER.csv
    ```
    * (Optionnel) Spécifiez un fichier de poids différent avec `--weights CHEMIN/VERS/POIDS.npz`. Par défaut, il utilise `Result/last_model_weights.npz`.
* Le programme chargera le modèle, le scaler, prétraitera les données d'entrée et affichera les prédictions ('B' ou 'M') pour chaque ligne.

## Tests

Pour lancer les tests unitaires, exécutez la commande suivante depuis le dossier racine (`MY_project/`) :

```bash
python3 -m unittest discover -s test -v
