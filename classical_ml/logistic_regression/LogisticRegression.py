import math


# σ(z) = 1 / (1 + e^-z)
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict(x, weights, bias):
    z = sum(w * xi for w, xi in zip(weights, x)) + bias
    return sigmoid(z)


def binary_loss(y_true, y_pred):
    return - (y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def train(X, y, lr=0.1, epochs=100):
    weights = [0.0 for _ in X[0]]
    bias = 0.0
    for epoch in range(epochs):
        dw = [0.0] * len(X[0])
        db = 0.0
        total_loss = 0.0

        for xi, yi in zip(X, y):
            y_pred = predict(xi, weights, bias)
            error = y_pred - yi

            # ∂L/∂w_j = (ŷ - y) * x_j
            for j in range(len(weights)):
                dw[j] += error * xi[j]

            # ∂L/∂b = ŷ - y
            db += error
            total_loss += binary_loss(yi, y_pred)

        # Update: w := w - η ∂L /∂w, b := b - η ∂L /∂b
        weights = [w - lr * d / len(X) for w, d in zip(weights, dw)]
        bias -= lr * db / len(X)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    return weights, bias


X_bin = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_bin = [0, 0, 0, 1]

# Train
w_bin, b_bin = train(X_bin, y_bin)

# Predict
print("\nBinary Predictions:")
for xi in X_bin:
    prob = predict(xi, w_bin, b_bin)
    print(f"Input {xi} => Predicted: {round(prob, 4)} → Class: {int(prob >= 0.5)}")
