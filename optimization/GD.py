import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


y = 1
x1 = 0
x2 = 1
y_pred = sigmoid(0.5)
error = y_pred - y

dw = error * x1
dw1 = error * x2
db = error
w = []
b = 0
lr = 0.01
w[0] = w[0] - lr * dw1
w[1] = w[1] - lr * dw
b = b - lr * db


# Function: f(w) = (w - 4)^2
def f(w):
    return (w - 4) ** 2


# Derivative: f'(w) = 2(w - 4)
def gradient(w):
    return 2 * (w - 4)


def gradient_descent(initial_w, lr=0.1, epochs=20):
    w = initial_w
    for i in range(epochs):
        gr = gradient(w)
        w = w - lr * gr
        print(f"Epoch {i + 1}: w = {w:.4f}, f(w) = {f(w):.4f}")
    return w


final_w = gradient_descent(initial_w=0)
print(f"\nFinal weight: {final_w:.4f}")

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, lr=0.01, epochs=1000):
    """
    Gradient Descent for Simple Linear Regression using NumPy.

    We aim to learn parameters w and b that minimize the Mean Squared Error (MSE):

        Loss = (1/n) * Σ (y_i - (w * x_i + b))^2

    Gradients:
        dL/dw = -(2/n) * Σ x_i * (y_i - ŷ_i)
        dL/db = -(2/n) * Σ (y_i - ŷ_i)
    
    Where:
        - ŷ_i = predicted output = w * x_i + b
        - y_i = true label
        - n = number of samples
        - lr = learning rate
    """

    n = len(X)
    w = 0.0  # Initial weight
    b = 0.0  # Initial bias

    for i in range(epochs):
        # Step 1: Make predictions
        # ŷ = w * x + b
        y_pred = w * X + b

        # Step 2: Compute the error
        # error = y - ŷ
        error = y - y_pred

        # Step 3: Compute gradients using the MSE derivative formulas
        # dw = -(2/n) * Σ x_i * (y_i - ŷ_i)
        # db = -(2/n) * Σ (y_i - ŷ_i)
        dw = -(2/n) * np.dot(X, error)
        db = -(2/n) * np.sum(error)

        # Step 4: Update parameters using gradient descent
        # w := w - lr * dw
        # b := b - lr * db
        w -= lr * dw
        b -= lr * db

        # Step 5: Print loss every 100 iterations
        # Loss = MSE = mean(error^2)
        if i % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {i:4d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

    return w, b

# Generate fake linear data: y = 2.5x + 1 + noise
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2.5 * X + 1.0 + np.random.randn(*X.shape) * 2  # adding noise

# Run gradient descent
w, b = gradient_descent(X, y, lr=0.01, epochs=1000)

# Final results
print(f"\nFinal weights — w: {w:.4f}, b: {b:.4f}")

# Plot result
plt.scatter(X, y, label='Data', alpha=0.6)
plt.plot(X, w * X + b, color='red', label='Fitted Line')
plt.title('Linear Regression with Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

from scipy.optimize import minimize
def mse_loss(params, X, y):
    w,b=params
    y_pred = w*X+b
    return np.mean((y-y_pred) ** 2)
res = minimize(mse_loss, x0=[0.0, 0.0], args=(X,y))
print(f"\n[scipy.optimize] Best w: {res.x[0]:.4f}, b: {res.x[1]:.4f}")
