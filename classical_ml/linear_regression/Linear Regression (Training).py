import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:
        # -2/N * (y - y_hat)^T * X[:, j]
        residuals = ground_truth - model_prediction
        feature_j = X[:, desired_weight]
        numerator = np.dot(residuals, feature_j)
        derivative = -2 * numerator / N 
        return derivative
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        W = np.array(initial_weights, dtype=np.float64, copy=True)
        N = len(X)
        y_true = np.squeeze(Y).astype(np.float32)
        for _ in range(num_iterations):
            y_pred = self.get_model_prediction(X, W)
            grad = np.zeros_like(W, dtype=np.float32)
            for j in range(W.shape[0]):
                grad[j] = self.get_derivative(y_pred, y_true, N, X, j)
            W = W - self.learning_rate * grad
        return np.round(W, 5)

X = [[1, 2, 3], [1, 1, 1]]
Y = [6, 3]
num_iterations = 10
initial_weights = [0.2, 0.1, 0.6]
sol = Solution()
sol.train_model(X, Y, num_iterations, initial_weights)