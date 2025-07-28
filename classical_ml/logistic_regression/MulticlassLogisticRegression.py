import math


# Softmax: P(y=k|x) = e^(z_k) / ∑ e^(z_j)
def softmax(z):
    exps = [math.exp(zi) for zi in z]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


# z_k = w_k^T x + b_k for each class
def predict_softmax(x, weights, biases):
    z = []
    for k in range(len(weights)):
        zk = sum(wi * xi for wi, xi in zip(weights[k], x)) + biases[k]
        z.append(zk)
    return softmax(z)


# Multiclass cross-entropy: -log(ŷ_true_class)
def multiclass_loss(y_true, y_probs):
    return -math.log(y_probs[y_true])


def train_multiclass(X, y, num_classes, lr=0.01, epochs=100):
    num_features = len(X[0])
    weights = [[0.0] * num_features for _ in range(num_classes)]
    biases = [0.0] * num_features
    for epoch in range(epochs):
        dw = [[0.0] * num_features for _ in range(num_classes)]
        db = [0.0] * num_classes
        total_loss = 0.0
        for xi, yi in zip(X, y):
            probs = predict_softmax(xi, weights, biases)
            total_loss += multiclass_loss(yi, probs)

            for k in range(num_classes):
                # ∂L/∂w_kj = (ŷ_k - y_k) * x_j
                error = probs[k] - (1 if k == yi else 0)
                for j in range(num_features):
                    dw[k][j] += error * xi[j]
                db[k] += error
        # Update: w := w - η ∂L/∂w, b := b - η ∂L/∂b
        for k in range(num_classes):
            for j in range(num_features):
                weights[k][j] -= lr * dw[k][j] / len(X)
            biases[k] -= lr * db[k] / len(X)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    return weights, biases


X_multi = [[1, 0], [0, 1], [1, 1]]
y_multi = [0, 1, 2]

# Train
w_multi, b_multi = train_multiclass(X_multi, y_multi, num_classes=3)

# Predict
print("\nMulticlass Predictions:")
for xi in X_multi:
    probs = predict_softmax(xi, w_multi, b_multi)
    predicted = probs.index(max(probs))
    print(f"Input {xi} => Class Probabilities: {[round(p, 4) for p in probs]} → Predicted Class: {predicted}")
