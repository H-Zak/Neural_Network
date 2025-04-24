import unittest
import numpy as np
# Assurez-vous que le chemin d'importation est correct depuis le dossier 'test'
# Si 'Classes' est au même niveau que 'test', cela pourrait nécessiter d'ajuster le PYTHONPATH
# ou d'utiliser des imports relatifs si vous structurez en package.
# Pour une exécution simple avec 'python -m unittest test/test.py' depuis la racine 'Amelioration_2',
# l'import suivant devrait fonctionner :
from Classes.MultilayerPerceptron import MultilayerPerceptron
import tempfile
import os

class TestActivationFunctions(unittest.TestCase):

    def setUp(self):
        """Met en place une instance du réseau pour accéder aux méthodes non statiques si nécessaire."""
        # On a besoin d'une instance pour appeler les méthodes, même si sigmoid pourrait être statique.
        # L'architecture ici n'a pas d'importance pour tester sigmoid seul.
        self.network = MultilayerPerceptron(layers=[2, 3, 2])

    def test_sigmoid_basic(self):
        """Teste la fonction sigmoid avec des valeurs scalaires."""
        self.assertAlmostEqual(self.network.sigmoid(0), 0.5)
        # Pour des valeurs grandes positives, sigmoid tend vers 1
        self.assertAlmostEqual(self.network.sigmoid(100), 1.0)
        # Pour des valeurs grandes négatives, sigmoid tend vers 0
        self.assertAlmostEqual(self.network.sigmoid(-100), 0.0)
        # Valeur intermédiaire
        # sigmoid(1) = 1 / (1 + exp(-1)) approx 0.7310585786
        self.assertAlmostEqual(self.network.sigmoid(1), 0.7310585786)

    def test_sigmoid_numpy_array(self):
        """Teste la fonction sigmoid avec un tableau NumPy."""
        input_array = np.array([0, 1, -100])
        expected_output = np.array([0.5, 0.7310585786, 0.0])
        output = self.network.sigmoid(input_array)
        # Vérifie si les tableaux sont presque égaux élément par élément
        np.testing.assert_almost_equal(output, expected_output)
    # --- Tests ReLU ---
    def test_relu_basic(self):
        """Teste la fonction ReLU avec des valeurs scalaires."""
        self.assertEqual(self.network.Relu(0), 0)
        self.assertEqual(self.network.Relu(10), 10)
        self.assertEqual(self.network.Relu(-10), 0)

    def test_relu_numpy_array(self):
        """Teste la fonction ReLU avec un tableau NumPy."""
        input_array = np.array([0, 10, -10, 5.5, -0.1])
        expected_output = np.array([0, 10, 0, 5.5, 0])
        output = self.network.Relu(input_array)
        np.testing.assert_array_equal(output, expected_output) # assert_array_equal pour les entiers/zéros exacts

    # --- Tests Dérivée ReLU ---
    def test_relu_derivative_basic(self):
        """Teste la dérivée de ReLU avec des valeurs scalaires."""
        self.assertEqual(self.network.relu_derivative(10), 1)
        self.assertEqual(self.network.relu_derivative(-10), 0)
        # Le cas Z=0 est souvent défini comme 0 ou 1. Votre implémentation donne 0.
        self.assertEqual(self.network.relu_derivative(0), 0)

    def test_relu_derivative_numpy_array(self):
        """Teste la dérivée de ReLU avec un tableau NumPy."""
        input_array = np.array([10, -10, 0, 0.1, -0.1])
        expected_output = np.array([1, 0, 0, 1, 0])
        output = self.network.relu_derivative(input_array)
        np.testing.assert_array_equal(output, expected_output)

    # --- Tests Dérivée Sigmoid ---
    def test_sigmoid_derivative_basic(self):
        """Teste la dérivée de Sigmoid avec des valeurs scalaires."""
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(self.network.sigmoid_derivative(0), 0.25)
        # Pour |Z| grand, la dérivée tend vers 0
        self.assertAlmostEqual(self.network.sigmoid_derivative(100), 0.0)
        self.assertAlmostEqual(self.network.sigmoid_derivative(-100), 0.0)

    def test_sigmoid_derivative_numpy_array(self):
         """Teste la dérivée de Sigmoid avec un tableau NumPy."""
         input_array = np.array([0, 100, -100])
         expected_output = np.array([0.25, 0.0, 0.0])
         output = self.network.sigmoid_derivative(input_array)
         np.testing.assert_almost_equal(output, expected_output, decimal=6) # Augmenter la précision si nécessaire

    # --- Tests Softmax ---
    def test_softmax_basic(self):
        """Teste la fonction softmax avec un vecteur simple."""
        input_array = np.array([[1], [2], [3]]) # Shape (3, 1) comme Z pour un exemple
        output = self.network.softmax(input_array)
        # Vérifier que la somme des probabilités est 1
        self.assertAlmostEqual(np.sum(output), 1.0)
        # Vérifier que les valeurs sont des probabilités (entre 0 et 1)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        # Vérifier que l'élément correspondant à l'entrée la plus élevée a la plus haute probabilité
        self.assertEqual(np.argmax(output), 2)

    def test_softmax_multiple_samples(self):
        """Teste la fonction softmax avec plusieurs échantillons (colonnes)."""
        input_array = np.array([[1, 10], [2, 5], [3, 1]]) # Shape (3, 2) - 3 classes, 2 exemples
        output = self.network.softmax(input_array)
        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (3, 2))
        # Vérifier que la somme sur l'axe des classes (axis=0) donne 1 pour chaque échantillon
        np.testing.assert_almost_equal(np.sum(output, axis=0), np.array([1.0, 1.0]))
        # Vérifier que les valeurs sont des probabilités
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        # Vérifier les classes prédites pour chaque échantillon
        self.assertTrue(np.array_equal(np.argmax(output, axis=0), np.array([2, 0])))


class TestCostFunction(unittest.TestCase):

    def setUp(self):
        """Met en place une instance du réseau."""
        # L'architecture exacte importe peu pour tester compute_cost seul
        self.network = MultilayerPerceptron(layers=[2, 3, 2]) # 2 classes en sortie

    def test_compute_cost_perfect_match(self):
        """Teste le coût quand la prédiction est quasi parfaite."""
        m = 2
        AL = np.array([[0.999, 0.001],
                       [0.001, 0.999]])
        Y = np.array([[1, 0],
                      [0, 1]])

        # Calculer la valeur attendue plus précisément
        expected_cost = -(1/m) * (1 * np.log(0.999) + 1 * np.log(0.999))
        # expected_cost sera approx 0.0010005

        cost = self.network.compute_cost(AL, Y)

        # Comparer le coût calculé à la valeur attendue calculée
        self.assertAlmostEqual(cost, expected_cost, places=6) # Augmenter 'places' pour plus de précision

        # Alternative : Vérifier juste si le coût est très petit
        # self.assertTrue(cost < 0.01, f"Le coût {cost} devrait être très proche de 0 pour une prédiction parfaite")
    def test_compute_cost_complete_mismatch(self):
        """Teste le coût quand la prédiction est très mauvaise."""
        m = 2
        AL = np.array([[0.001, 0.999],
                       [0.999, 0.001]])
        Y = np.array([[1, 0],
                      [0, 1]])

        # Le coût attendu est élevé :
        # -(1/2) * [ (1*log(0.001) + 0*log(0.999)) + (0*log(0.999) + 1*log(0.001)) ]
        # = -(1/2) * [ log(0.001) + log(0.001) ]
        # = -log(0.001) approx -(-6.907) = 6.907
        expected_cost = -np.log(0.001)
        cost = self.network.compute_cost(AL, Y)
        self.assertAlmostEqual(cost, expected_cost, places=3)

    def test_compute_cost_intermediate(self):
        """Teste le coût avec des valeurs intermédiaires."""
        m = 3
        # Prédictions (probabilités)
        AL = np.array([[0.7, 0.4, 0.1],  # Classe 0
                       [0.3, 0.6, 0.9]]) # Classe 1
        # Vrais labels (one-hot)
        Y = np.array([[1, 0, 0],      # Échantillon 1 = Classe 0
                      [0, 1, 1]])     # Échantillons 2, 3 = Classe 1

        # Coût attendu = -(1/3) * [ (1*log(0.7)+0*log(0.3)) + (0*log(0.4)+1*log(0.6)) + (0*log(0.1)+1*log(0.9)) ]
        # = -(1/3) * [ log(0.7) + log(0.6) + log(0.9) ]
        expected_cost = -(1/3) * (np.log(0.7) + np.log(0.6) + np.log(0.9))
        cost = self.network.compute_cost(AL, Y)
        self.assertAlmostEqual(cost, expected_cost, places=5)

    def test_compute_cost_clipping(self):
        """Teste si le clipping évite log(0)."""
        m = 1
        # Prédiction exacte de 0 ou 1 devrait être clippée à epsilon ou 1-epsilon
        AL = np.array([[1.0], [0.0]])
        Y = np.array([[1], [0]])
        # Sans clipping, log(0) donnerait -inf. Avec clipping, on calcule log(1-epsilon) ou log(epsilon).
        # La fonction compute_cost contient np.clip(AL, epsilon, 1 - epsilon)
        try:
            cost = self.network.compute_cost(AL, Y)
            # Le coût doit être un nombre fini (proche de 0 si epsilon petit)
            self.assertTrue(np.isfinite(cost))
            self.assertGreaterEqual(cost, 0.0) # Le coût doit être >= 0
        except Exception as e:
            self.fail(f"compute_cost a levé une exception inattendue : {e}")

# ... (Imports et classes TestActivationFunctions, TestCostFunction existants) ...

# === NOUVELLE CLASSE DE TEST ===
class TestInitialization(unittest.TestCase):

    def test_initialise_value_shapes(self):
        """Vérifie les dimensions des poids et biais après initialisation."""
        layers_config = [5, 10, 8, 2] # Exemple d'architecture : 5 entrées, 2 couches cachées, 2 sorties
        network = MultilayerPerceptron(layers=layers_config)
        # initialise_value() est appelé dans le __init__, les poids/biais devraient exister

        # 1. Vérifier le nombre de matrices de poids et de biais
        expected_num_matrices = len(layers_config) - 1
        self.assertEqual(len(network.weight), expected_num_matrices)
        self.assertEqual(len(network.biais), expected_num_matrices)

        # 2. Vérifier les dimensions de chaque matrice
        for i in range(expected_num_matrices):
            # La couche actuelle est i+1, la précédente est i
            n_neurons_current = layers_config[i+1]
            n_neurons_previous = layers_config[i]

            expected_weight_shape = (n_neurons_current, n_neurons_previous)
            expected_bias_shape = (n_neurons_current, 1)

            self.assertEqual(network.weight[i].shape, expected_weight_shape,
                             f"Dimension incorrecte pour weight[{i}]")
            self.assertEqual(network.biais[i].shape, expected_bias_shape,
                             f"Dimension incorrecte pour biais[{i}]")

            # Optionnel : Vérifier que les biais sont initialisés à zéro
            np.testing.assert_array_equal(network.biais[i], np.zeros(expected_bias_shape),
                                         f"Biais[{i}] non initialisé à zéro.")

    def test_initialise_value_different_config(self):
        """Vérifie les dimensions avec une autre architecture."""
        layers_config = [30, 50, 2] # Une seule couche cachée
        network = MultilayerPerceptron(layers=layers_config)

        expected_num_matrices = len(layers_config) - 1
        self.assertEqual(len(network.weight), expected_num_matrices)
        self.assertEqual(len(network.biais), expected_num_matrices)

        # Couche 1 (indice 0)
        self.assertEqual(network.weight[0].shape, (50, 30))
        self.assertEqual(network.biais[0].shape, (50, 1))
        np.testing.assert_array_equal(network.biais[0], np.zeros((50, 1)))

        # Couche 2 (indice 1)
        self.assertEqual(network.weight[1].shape, (2, 50))
        self.assertEqual(network.biais[1].shape, (2, 1))
        np.testing.assert_array_equal(network.biais[1], np.zeros((2, 1)))


# ... (Imports et classes TestActivationFunctions, TestCostFunction existants) ...

# === NOUVELLE CLASSE DE TEST ===
class TestInitialization(unittest.TestCase):

    def test_initialise_value_shapes(self):
        """Vérifie les dimensions des poids et biais après initialisation."""
        layers_config = [5, 10, 8, 2] # Exemple d'architecture : 5 entrées, 2 couches cachées, 2 sorties
        network = MultilayerPerceptron(layers=layers_config)
        # initialise_value() est appelé dans le __init__, les poids/biais devraient exister

        # 1. Vérifier le nombre de matrices de poids et de biais
        expected_num_matrices = len(layers_config) - 1
        self.assertEqual(len(network.weight), expected_num_matrices)
        self.assertEqual(len(network.biais), expected_num_matrices)

        # 2. Vérifier les dimensions de chaque matrice
        for i in range(expected_num_matrices):
            # La couche actuelle est i+1, la précédente est i
            n_neurons_current = layers_config[i+1]
            n_neurons_previous = layers_config[i]

            expected_weight_shape = (n_neurons_current, n_neurons_previous)
            expected_bias_shape = (n_neurons_current, 1)

            self.assertEqual(network.weight[i].shape, expected_weight_shape,
                             f"Dimension incorrecte pour weight[{i}]")
            self.assertEqual(network.biais[i].shape, expected_bias_shape,
                             f"Dimension incorrecte pour biais[{i}]")

            # Optionnel : Vérifier que les biais sont initialisés à zéro
            np.testing.assert_array_equal(network.biais[i], np.zeros(expected_bias_shape),
                                         f"Biais[{i}] non initialisé à zéro.")

    def test_initialise_value_different_config(self):
        """Vérifie les dimensions avec une autre architecture."""
        layers_config = [30, 50, 2] # Une seule couche cachée
        network = MultilayerPerceptron(layers=layers_config)

        expected_num_matrices = len(layers_config) - 1
        self.assertEqual(len(network.weight), expected_num_matrices)
        self.assertEqual(len(network.biais), expected_num_matrices)

        # Couche 1 (indice 0)
        self.assertEqual(network.weight[0].shape, (50, 30))
        self.assertEqual(network.biais[0].shape, (50, 1))
        np.testing.assert_array_equal(network.biais[0], np.zeros((50, 1)))

        # Couche 2 (indice 1)
        self.assertEqual(network.weight[1].shape, (2, 50))
        self.assertEqual(network.biais[1].shape, (2, 1))
        np.testing.assert_array_equal(network.biais[1], np.zeros((2, 1)))

# ... (Classes de test existantes) ...

# === NOUVELLE CLASSE DE TEST ===
class TestSaveLoadWeights(unittest.TestCase):

    def test_save_load_cycle(self):
        """Vérifie que les poids sauvegardés puis chargés sont identiques."""
        layers_config = [4, 8, 3, 2] # Une architecture exemple
        network_original = MultilayerPerceptron(layers=layers_config)
        # Remplir avec des poids/biais spécifiques ou laisser l'initialisation aléatoire

        # Créer un dossier temporaire pour le test
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_weights.npz")

            # 1. Sauvegarder les poids du réseau original
            network_original.save_weights(filepath)
            self.assertTrue(os.path.exists(filepath)) # Vérifier que le fichier a été créé

            # 2. Créer un nouveau réseau avec la même architecture
            network_loaded = MultilayerPerceptron(layers=layers_config)
            # Vérifier qu'ils ne sont pas identiques AVANT chargement (très improbable mais bon)
            self.assertFalse(np.array_equal(network_original.weight[0], network_loaded.weight[0]))

            # 3. Charger les poids sauvegardés dans le nouveau réseau
            network_loaded.load_weights(filepath)

            # 4. Vérifier que les poids et biais sont maintenant identiques
            self.assertEqual(len(network_original.weight), len(network_loaded.weight))
            self.assertEqual(len(network_original.biais), len(network_loaded.biais))

            for i in range(len(network_original.weight)):
                np.testing.assert_array_equal(network_original.weight[i], network_loaded.weight[i],
                                             f"Les poids de la couche {i} diffèrent après chargement.")
                np.testing.assert_array_equal(network_original.biais[i], network_loaded.biais[i],
                                             f"Les biais de la couche {i} diffèrent après chargement.")

    def test_load_architecture_mismatch(self):
        """Vérifie qu'une erreur est levée si l'architecture ne correspond pas."""
        layers_config1 = [4, 8, 2]
        layers_config2 = [4, 10, 2] # Architecture différente
        network1 = MultilayerPerceptron(layers=layers_config1)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_weights_arch1.npz")
            network1.save_weights(filepath) # Sauvegarde l'architecture [4, 8, 2]

            # Essayer de charger dans un réseau avec une architecture différente
            network2 = MultilayerPerceptron(layers=layers_config2)
            # S'attendre à une ValueError lors du chargement à cause de l'incompatibilité
            with self.assertRaises(ValueError):
                network2.load_weights(filepath)


if __name__ == "__main__":
    unittest.main()
