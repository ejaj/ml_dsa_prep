import numpy as np


class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        self.loss_scale = float(loss_scale)
        self._overflow = False

    def forward(self, weights, inputs, targets):
        x16 = np.asarray(inputs, dtype=np.float16)
        w16 = np.asarray(weights, dtype=np.float16)
        t16 = np.asarray(targets, dtype=np.float16)

        preds16 = x16 @ w16
        diff16 = preds16 - t16
        mse16 = np.mean(diff16 * diff16, dtype=np.float16)

        scaled32 = np.float32(mse16) * np.float32(self.loss_scale)
        return float(scaled32)

    def backward(self, gradients):
        g32 = np.asarray(gradients, dtype=np.float32)
        inv_scale = np.float32(1.0 / self.loss_scale)
        unscaled = g32 * inv_scale

        if not np.all(np.isfinite(unscaled)):
            self._overflow = True
            return np.zeros_like(unscaled, dtype=np.float32)
        self._overflow = False
        return unscaled


# Create an instance of the MixedPrecision class
mp = MixedPrecision(loss_scale=1024.0)

# Define model parameters (weights), inputs, and targets
weights = np.array([0.5, -0.3], dtype=np.float32)
inputs = np.array([[1.0, 2.0],
                   [3.0, 4.0]], dtype=np.float32)
targets = np.array([1.0, 0.0], dtype=np.float32)

# 1️⃣ Forward pass: compute scaled loss
loss = mp.forward(weights, inputs, targets)
print(f"Loss: {loss:.4f}")
print(f"Loss dtype: {type(loss).__name__}")

# 2️⃣ Backward pass: unscale gradients and check for overflow
grads = np.array([512.0, -256.0], dtype=np.float32)
result = mp.backward(grads)
print(f"Gradients: {result}")
print(f"Grad dtype: {result.dtype}")
