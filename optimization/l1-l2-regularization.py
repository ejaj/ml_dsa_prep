import numpy as np

np.random.seed(42)
X = np.random.rand(100, 1) * 10
print(X.shape)
y = 3 * X.squeeze() + 7 + np.random.randn(100)  # y = 3x + 7 + noise
print(y.shape)
# Add bias term to X (intercept)
X_b = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (100, 2)


# Gradient descent parameters
epochs = 1000
lr = 0.01
lambda_l1 = 0.1  # L1 strength
lambda_l2 = 0.1  # L2 strength

# Initialize weights for all 3 cases
w_plain = np.random.randn(2)
w_l2 = np.random.randn(2)
w_l1 = np.random.randn(2)
print(w_plain)

for epoc in range(epochs):
    # Prediction
    y_pred_plain = X_b.dot(w_plain)
    y_pred_l2 = X_b.dot(w_l2)
    y_pred_l1 = X_b.dot(w_l1)

    # Gradients
    grad_plain = -2 * X_b.T.dot(y-y_pred_plain) / len(y)
    grad_l2 = -2 * X_b.T.dot(y-y_pred_l2) / len(y) + 2*lambda_l2*w_l2
    grad_l1 = -2 * X_b.T.dot(y-y_pred_l1) / len(y) + lambda_l1 * np.sign(w_l1)

    # Update Weights
    w_plain -= lr * grad_plain
    w_l2 -= lr * grad_l2
    w_l1 -= lr * grad_l1

print("No Regularization weights:", w_plain)
print("L2 Regularization weights:", w_l2)
print("L1 Regularization weights:", w_l1)














