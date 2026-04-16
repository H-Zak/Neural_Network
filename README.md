# Multilayer Perceptron

Implémentation from scratch d'un perceptron multicouche pour classifier des tumeurs du sein (Wisconsin Breast Cancer dataset) en bénignes ou malignes.

Sujet 42 — Machine Learning project, version 5.1.

## Lancer

```bash
make install

# 1. Séparer le dataset (80% train / 20% validation)
python3 -m neural_network.main --mode split

# 2. Entraîner (interactif : époques, learning rate, batch, couches, early stopping)
python3 -m neural_network.main --mode train

# 3. Prédire sur de nouvelles données
python3 -m neural_network.main --mode predict --input dataset/data_test.csv
python3 -m neural_network.main --mode predict --input dataset/data_cancer_1.csv
```

Le CSV de prédiction peut être au format complet (ID + label + 30 features), avec label seul, ou features seules — le format est détecté automatiquement. Si les labels sont présents, la binary cross-entropy et l'accuracy sont affichées.

## Architecture

- Couches cachées configurables (minimum 2), activation sigmoid
- Couche de sortie softmax (2 classes)
- Loss : cross-entropy catégorielle (training) / binary cross-entropy (évaluation)
- Optimisation : gradient descent (full-batch ou mini-batch)
- Preprocessing : StandardScaler (fit sur train, transform sur validation/test)
- Split : 80% train / 20% validation (stratifié)
- Initialisation : Xavier (sqrt(1/n))
- Early stopping avec patience configurable

## Structure

```
├── Classes/
│   ├── MultilayerPerceptron.py   # Réseau de neurones
│   └── data.py                   # Chargement et preprocessing
├── neural_network/
│   ├── main.py                   # Point d'entrée (split / train / predict)
│   └── parsing.py                # Input utilisateur + graphes
├── test/
│   └── test.py                   # Tests unitaires (30 tests)
├── dataset/
│   ├── data_cancer.csv           # Dataset principal (569 samples)
│   ├── data_cancer_1.csv         # Données test (6 samples)
│   └── data_test.csv             # Données test (3 samples)
├── Result/                       # Poids, scaler, graphes, performances
├── Makefile
└── requirement.txt
```

## Tests

```bash
make test
# ou
python3 -m unittest discover -s test -v
```

## Auteurs

Zhamdouc, dnieto-c
