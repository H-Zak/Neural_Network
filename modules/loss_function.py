import numpy as np 

# Docs : If any of the predicted values in predicted_values is exactly 0 or 1, then np.log(0) will give an undefined value (-inf).
# To avoid this problem, a small value (epsilon) is usually added to maintain numerical stability and prevent the logarithm from dealing with 0

def binary_cross_entropy(predicted_values: np.ndarray, real_values: np.ndarray) -> float:
    """
    Compute the binary cross-entropy loss between predicted values and real values.

    Args:
    predicted_values (np.ndarray): Predicted probabilities (values between 0 and 1).
    real_values (np.ndarray): Actual binary labels (0 or 1).

    Returns:
    float: The binary cross-entropy loss.
    """
    # Add an optional check to make sure that the predicted_values and actual_values dimensions match.
    assert predicted_values.shape == real_values.shape, "Shape mismatch between predicted and real values."

    training_examples = real_values.shape[0]
    
    # Prevent log(0) issues by clipping the predicted values
    epsilon = 1e-15
    predicted_values = np.clip(predicted_values, epsilon, 1 - epsilon)
    
    # Compute the binary cross-entropy
    errors = (real_values * np.log(predicted_values)) + ((1 - real_values) * np.log(1 - predicted_values))
    sum_errors = (-1 / training_examples) * np.sum(errors)

    return sum_errors


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the cross-entropy loss between predicted values and one-hot encoded labels.

    Args:
    y_pred (np.ndarray): Predicted probabilities from softmax (shape: [num_classes, batch_size]).
    y_true (np.ndarray): One-hot encoded actual labels (shape: [num_classes, batch_size]).

    Returns:
    float: The cross-entropy loss.
    """
    assert y_pred.shape == y_true.shape, "Shape mismatch between predicted and real values."

    # Avoid log(0) by clipping the predicted values
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy loss per example and average
    loss_per_example = -np.sum(y_true * np.log(y_pred), axis=0)
    loss = np.mean(loss_per_example)
    return loss
