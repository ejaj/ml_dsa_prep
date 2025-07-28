import numpy as np
import matplotlib.pyplot as plt

class SimpleSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=100):
        # lr = learning rate (η)
        # lambda_param = regularization strength (λ) = 1/C
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        # Initialize w = 0, b = 0
        self.w = np.zeros(n_features)
        self.b = 0
        # Gradient Descent Loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the margin: y_i (w·x_i + b)
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # If y_i(w·x_i + b) ≥ 1:
                    # No hinge loss → only regularization term in gradient
                    # ∇w = λw ; ∇b = 0
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    # If y_i(w·x_i + b) < 1:
                    # Hinge loss active → update using:
                    # ∇w = λw - y_i·x_i
                    # ∇b = -y_i
                    dw = self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]
                # Gradient descent update:
                # w ← w - η ∇w
                # b ← b - η ∇b
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
    def predict(self, X):
        # Prediction = sign(w·x + b)
        return np.sign(np.dot(X, self.w) + self.b)
    
    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

        # Decision boundary: w·x + b = 0 → x2 = -(w1·x1 + b)/w2
        x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y_vals = -(self.w[0] * x_vals + self.b) / self.w[1]

        # Margin = 1 / ||w||
        margin = 1 / np.linalg.norm(self.w)
        y_margin_up = y_vals + margin
        y_margin_down = y_vals - margin

        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
        plt.plot(x_vals, y_margin_up, 'g--', label='Margin +1')
        plt.plot(x_vals, y_margin_down, 'g--', label='Margin -1')
        plt.title("SVM from Scratch (with Gradient Descent)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)
        plt.show()

X_pos = np.array([[2, 3], [3, 4], [4, 5]])
X_neg = np.array([[6, 5], [7, 6], [8, 8]])
X = np.vstack((X_pos, X_neg))
y = np.array([1, 1, 1, -1, -1, -1])  # class labels

model = SimpleSVM(lr=0.001, lambda_param=0.01, n_iters=100)
model.fit(X, y)

model.plot_decision_boundary(X, y)

x_test = np.array([[5, 5]])
pred = model.predict(x_test)
print(f"Prediction for {x_test[0]}: {int(pred[0])}")


