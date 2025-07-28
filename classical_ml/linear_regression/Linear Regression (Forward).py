import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the linear regression model prediction using the formula:

            y_pred = X * w

        where:
            - X is the input feature matrix (Nx3)
            - w is the weight vector (3x1)
            - y_pred is the predicted output (Nx1)

        Args:
            X: Nx3 NumPy array representing the dataset.
            weights: 3x1 NumPy array representing the model weights.
        Returns:
            Nx1 NumPy array with predicted values, rounded to 5 decimal places.
        """
        predictions = np.matmul(X, weights)
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        """
        Computes the mean squared error (MSE) using the formula:

            MSE = (1/n) * Σ (y_true - y_pred)^2

        where:
            - y_true is the actual ground truth values (Nx1)
            - y_pred is the model's predicted values (Nx1)
            - n is the number of samples

        Args:
            model_prediction: Nx1 NumPy array with model predictions.
            ground_truth: Nx1 NumPy array with true values.
        Returns:
            Mean squared error, rounded to 5 decimal places.
        """
        mse = np.mean(np.square(model_prediction - ground_truth))  # Compute MSE (Σ (y_true - y_pred)^2 / n)
        return np.round(mse, 5)
