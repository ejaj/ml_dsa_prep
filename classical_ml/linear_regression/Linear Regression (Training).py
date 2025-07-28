import numpy as np


class LinearRegression:

    def get_model_prediction(self, X, weights):
        """
        Computes the linear regression model prediction using:
        y_pred = X * w
        """
        return np.matmul(X, weights)  # Compute predictions

    def get_error(self, model_prediction, ground_truth):
        """
        Computes Mean Squared Error (MSE):

        MSE = (1/n) * Σ (y_true - y_pred)^2
        """
        mse = np.mean(np.square(model_prediction - ground_truth))
        return round(mse, 5)

    def get_derivative(self, model_prediction, ground_truth, N, X, desired_weight):
        """
        Computes the derivative of the loss function (MSE) w.r.t. weight w_j:
                ∂J/∂w_j = - (2/N) * Σ (y_i - y_pred_i) * x_ij
        """
        derivative = -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N
        return derivative

    def train_model(self, X, Y, num_iterations, initial_weights):
        """
        Trains the linear regression model using gradient descent:

            w_j = w_j - α * (∂J/∂w_j)
        """
        weights = np.array(initial_weights, dtype=np.float64)
        learning_rate = 0.01
        N = len(X)

        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, weights)
            for j in range(len(weights)):  # Update each weight separately
                gradient = self.get_derivative(predictions, Y, N, X, j)
                weights[j] -= learning_rate * gradient
        return np.round(weights, 5)


X = np.array([
    [1, 2, 3],
    [1, 1, 1]
], dtype=np.float64)

Y = np.array([6, 3], dtype=np.float64)  # Ground truth values
num_iterations = 10
initial_weights = np.array([0.2, 0.1, 0.6], dtype=np.float64)

solution = LinearRegression()
final_weights = solution.train_model(X, Y, num_iterations, initial_weights)
print("Final Weights:", final_weights)
