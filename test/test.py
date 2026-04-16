import unittest
import numpy as np
import tempfile
import os
from Classes.MultilayerPerceptron import MultilayerPerceptron


class TestActivationFunctions(unittest.TestCase):

    def setUp(self):
        self.network = MultilayerPerceptron(layers=[2, 3, 2])

    def test_sigmoid_basic(self):
        self.assertAlmostEqual(self.network.sigmoid(0), 0.5)
        self.assertAlmostEqual(self.network.sigmoid(100), 1.0)
        self.assertAlmostEqual(self.network.sigmoid(-100), 0.0)
        self.assertAlmostEqual(self.network.sigmoid(1), 0.7310585786)

    def test_sigmoid_numpy_array(self):
        output = self.network.sigmoid(np.array([0, 1, -100]))
        np.testing.assert_almost_equal(output, np.array([0.5, 0.7310585786, 0.0]))

    def test_sigmoid_derivative_basic(self):
        self.assertAlmostEqual(self.network.sigmoid_derivative(0), 0.25)
        self.assertAlmostEqual(self.network.sigmoid_derivative(100), 0.0)
        self.assertAlmostEqual(self.network.sigmoid_derivative(-100), 0.0)

    def test_sigmoid_derivative_numpy_array(self):
        output = self.network.sigmoid_derivative(np.array([0, 100, -100]))
        np.testing.assert_almost_equal(output, np.array([0.25, 0.0, 0.0]), decimal=6)

    def test_relu_basic(self):
        self.assertEqual(self.network.relu(0), 0)
        self.assertEqual(self.network.relu(10), 10)
        self.assertEqual(self.network.relu(-10), 0)

    def test_relu_numpy_array(self):
        output = self.network.relu(np.array([0, 10, -10, 5.5, -0.1]))
        np.testing.assert_array_equal(output, np.array([0, 10, 0, 5.5, 0]))

    def test_relu_derivative_basic(self):
        self.assertEqual(self.network.relu_derivative(10), 1)
        self.assertEqual(self.network.relu_derivative(-10), 0)
        self.assertEqual(self.network.relu_derivative(0), 0)

    def test_relu_derivative_numpy_array(self):
        output = self.network.relu_derivative(np.array([10, -10, 0, 0.1, -0.1]))
        np.testing.assert_array_equal(output, np.array([1, 0, 0, 1, 0]))

    def test_softmax_sums_to_one(self):
        output = self.network.softmax(np.array([[1], [2], [3]]))
        self.assertAlmostEqual(np.sum(output), 1.0)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        self.assertEqual(np.argmax(output), 2)

    def test_softmax_multiple_samples(self):
        output = self.network.softmax(np.array([[1, 10], [2, 5], [3, 1]]))
        self.assertEqual(output.shape, (3, 2))
        np.testing.assert_almost_equal(np.sum(output, axis=0), np.array([1.0, 1.0]))
        self.assertTrue(np.array_equal(np.argmax(output, axis=0), np.array([2, 0])))

    def test_softmax_extreme_values_no_nan(self):
        output = self.network.softmax(np.array([[1000, 0], [0, 0]]))
        self.assertFalse(np.any(np.isnan(output)))
        np.testing.assert_almost_equal(np.sum(output, axis=0), np.array([1.0, 1.0]))


class TestCostFunction(unittest.TestCase):

    def setUp(self):
        self.network = MultilayerPerceptron(layers=[2, 3, 2])

    def test_cost_perfect_prediction(self):
        AL = np.array([[0.999, 0.001], [0.001, 0.999]])
        Y = np.array([[1, 0], [0, 1]])
        expected = -(1 / 2) * (np.log(0.999) + np.log(0.999))
        self.assertAlmostEqual(self.network.compute_cost(AL, Y), expected, places=6)

    def test_cost_bad_prediction(self):
        AL = np.array([[0.001, 0.999], [0.999, 0.001]])
        Y = np.array([[1, 0], [0, 1]])
        expected = -np.log(0.001)
        self.assertAlmostEqual(self.network.compute_cost(AL, Y), expected, places=3)

    def test_cost_intermediate(self):
        AL = np.array([[0.7, 0.4, 0.1], [0.3, 0.6, 0.9]])
        Y = np.array([[1, 0, 0], [0, 1, 1]])
        expected = -(1 / 3) * (np.log(0.7) + np.log(0.6) + np.log(0.9))
        self.assertAlmostEqual(self.network.compute_cost(AL, Y), expected, places=5)

    def test_cost_clipping_prevents_log_zero(self):
        AL = np.array([[1.0], [0.0]])
        Y = np.array([[1], [0]])
        cost = self.network.compute_cost(AL, Y)
        self.assertTrue(np.isfinite(cost))
        self.assertGreaterEqual(cost, 0.0)


class TestXavierInitialization(unittest.TestCase):

    def test_weight_shapes(self):
        layers = [5, 10, 8, 2]
        network = MultilayerPerceptron(layers=layers)

        self.assertEqual(len(network.weight), 3)
        self.assertEqual(len(network.biais), 3)

        for i in range(len(layers) - 1):
            self.assertEqual(network.weight[i].shape, (layers[i + 1], layers[i]))
            self.assertEqual(network.biais[i].shape, (layers[i + 1], 1))
            np.testing.assert_array_equal(network.biais[i], np.zeros((layers[i + 1], 1)))

    def test_xavier_scaling(self):
        np.random.seed(42)
        layers = [1000, 500, 2]
        network = MultilayerPerceptron(layers=layers)
        expected_std = np.sqrt(1 / 1000)
        actual_std = np.std(network.weight[0])
        self.assertAlmostEqual(actual_std, expected_std, places=2)

    def test_constructor_rejects_invalid_layers(self):
        with self.assertRaises(ValueError):
            MultilayerPerceptron(layers="invalid")
        with self.assertRaises(ValueError):
            MultilayerPerceptron(layers=[5])
        with self.assertRaises(ValueError):
            MultilayerPerceptron(layers=[5, -1, 2])
        with self.assertRaises(ValueError):
            MultilayerPerceptron(layers=[5, 0, 2])


class TestForwardPropagation(unittest.TestCase):

    def test_output_shape(self):
        network = MultilayerPerceptron(layers=[4, 8, 2])
        X = np.random.randn(4, 10)
        output, Zs, activations = network.forward_propagation(X)

        self.assertEqual(output.shape, (2, 10))
        self.assertEqual(len(Zs), 2)
        self.assertEqual(len(activations), 3)

    def test_output_is_probability(self):
        network = MultilayerPerceptron(layers=[4, 8, 2])
        X = np.random.randn(4, 5)
        output, _, _ = network.forward_propagation(X)

        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
        np.testing.assert_almost_equal(np.sum(output, axis=0), np.ones(5))


class TestPrediction(unittest.TestCase):

    def test_predict_returns_one_hot(self):
        network = MultilayerPerceptron(layers=[4, 8, 2])
        X = np.random.randn(4, 5)
        pred = network.predict(X)

        self.assertEqual(pred.shape, (2, 5))
        np.testing.assert_array_equal(np.sum(pred, axis=0), np.ones(5))
        self.assertTrue(np.all((pred == 0) | (pred == 1)))

    def test_predict_matches_forward(self):
        network = MultilayerPerceptron(layers=[4, 6, 2])
        X = np.random.randn(4, 3)
        output, _, _ = network.forward_propagation(X)
        pred = network.predict(X)

        expected_classes = np.argmax(output, axis=0)
        pred_classes = np.argmax(pred, axis=0)
        np.testing.assert_array_equal(pred_classes, expected_classes)


class TestMiniBatches(unittest.TestCase):

    def test_mini_batch_covers_all_samples(self):
        network = MultilayerPerceptron(layers=[3, 4, 2])
        X = np.random.randn(3, 10)
        Y = np.random.randn(2, 10)
        batches = network.create_mini_batches(X, Y, 3)

        total_samples = sum(b[0].shape[1] for b in batches)
        self.assertEqual(total_samples, 10)

    def test_mini_batch_shapes(self):
        network = MultilayerPerceptron(layers=[3, 4, 2])
        X = np.random.randn(3, 10)
        Y = np.random.randn(2, 10)
        batches = network.create_mini_batches(X, Y, 4)

        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0].shape, (3, 4))
        self.assertEqual(batches[0][1].shape, (2, 4))
        self.assertEqual(batches[2][0].shape, (3, 2))


class TestTrainingConvergence(unittest.TestCase):

    def test_xor_converges(self):
        np.random.seed(42)
        network = MultilayerPerceptron(layers=[2, 8, 2])

        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        Y = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

        network.train(X, Y.T, num_epochs=5000, learning_rate=0.5, x_val=X, y_val=Y.T)

        self.assertTrue(len(network.costs) > 1)
        self.assertLess(network.costs[-1], network.costs[0])

    def test_cost_decreases(self):
        np.random.seed(42)
        network = MultilayerPerceptron(layers=[3, 6, 2])

        X = np.random.randn(3, 20)
        raw_labels = (X[0] > 0).astype(int)
        Y = np.zeros((20, 2))
        Y[np.arange(20), raw_labels] = 1

        network.train(X, Y, num_epochs=2000, learning_rate=0.1, x_val=X, y_val=Y)

        self.assertLess(network.costs[-1], network.costs[0])


class TestEarlyStopping(unittest.TestCase):

    def test_early_stopping_triggers_on_overfitting(self):
        np.random.seed(0)
        network = MultilayerPerceptron(layers=[2, 20, 2])

        X_train = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        Y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # (N, 2)

        X_val = np.random.randn(2, 4)
        Y_val = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # (N, 2)

        network.train(X_train, Y_train, num_epochs=50000, learning_rate=0.5,
                      x_val=X_val, y_val=Y_val, patience=5)

        max_checks = 50000 // 100
        self.assertLess(len(network.costs), max_checks)

    def test_no_early_stopping_when_patience_zero(self):
        np.random.seed(42)
        network = MultilayerPerceptron(layers=[2, 4, 2])

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[1, 0], [0, 1]])

        network.train(X, Y, num_epochs=100, learning_rate=0.1,
                      x_val=X, y_val=Y, patience=0)

        self.assertEqual(len(network.costs), 100)

    def test_early_stopping_restores_best_weights(self):
        np.random.seed(0)
        network = MultilayerPerceptron(layers=[2, 20, 2])

        X_train = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        Y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # (N, 2)
        X_val = np.random.randn(2, 4)
        Y_val = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # (N, 2)

        network.train(X_train, Y_train, num_epochs=50000, learning_rate=0.5,
                      x_val=X_val, y_val=Y_val, patience=5)

        final_val_cost = network.function_valid_cost(X_val, Y_val.T)
        self.assertLessEqual(final_val_cost, network.validation_cost[0])


class TestSaveLoadWeights(unittest.TestCase):

    def test_save_load_preserves_weights(self):
        original = MultilayerPerceptron(layers=[4, 8, 3, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_weights.npz")
            original.save_weights(filepath)
            self.assertTrue(os.path.exists(filepath))

            loaded = MultilayerPerceptron(layers=[4, 8, 3, 2])
            loaded.load_weights(filepath)

            for i in range(len(original.weight)):
                np.testing.assert_array_equal(original.weight[i], loaded.weight[i])
                np.testing.assert_array_equal(original.biais[i], loaded.biais[i])

    def test_load_rejects_wrong_architecture(self):
        network1 = MultilayerPerceptron(layers=[4, 8, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_weights.npz")
            network1.save_weights(filepath)

            network2 = MultilayerPerceptron(layers=[4, 10, 2])
            with self.assertRaises(ValueError):
                network2.load_weights(filepath)


if __name__ == "__main__":
    unittest.main()
