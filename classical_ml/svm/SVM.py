import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) Classifier - Implemented from Scratch

    📌 This implementation uses **Gradient Descent** to solve the optimization problem.

    🎯 **Mathematical Formulation:**
    - Given a dataset (X, y), SVM finds the hyperplane that **maximizes the margin**.
    - The equation of the **decision boundary (hyperplane)** is:

      w · x + b = 0

    - **Optimization Problem (Hard Margin SVM)**:
      
      Minimize:
      𝐿(𝑤) =  1/2 ||w||²  
      
      Subject to:
      yᵢ (w · xᵢ + b) ≥ 1 for all i

    - **Soft Margin SVM (Allowing Some Errors)**:
      
      Minimize:
      𝐿(𝑤) = 1/2 ||w||² + C Σ ξᵢ

      Subject to:
      yᵢ (w · xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

      - C: Trade-off between margin maximization and misclassification.
      - ξᵢ: Slack variable for misclassified points.

    - The **weight vector (w) and bias (b)** are updated using **Gradient Descent**.
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        """
        Initializes the SVM model with hyperparameters.

        :param learning_rate: Step size for weight updates (default: 0.01).
        :param lambda_param: Regularization parameter to prevent overfitting (default: 0.01).
        :param n_iters: Number of iterations for gradient descent (default: 1000).
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None # Weight vector
        self.b = None # Bias term
    def fit(self, X, y):
        """
        Trains the SVM model using gradient descent.

        :param X: Feature matrix of shape (n_samples, n_features).
        :param y: Target labels (1 or -1).
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        # Convert labels from {0,1} to {-1,1} for hinge loss calculation
        y = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1
                if condition:
                    # Correctly classified, only apply weight decay
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified, apply hinge loss penalty
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X[i], y[i]))
                    self.b -= self.lr * y[i]
    def predict(self, X):
        """
        Predicts class labels for given test data.

        :param X: Feature matrix of shape (n_samples, n_features).
        :return: Predicted labels (1 or -1).
        """
        return np.sign(np.dot(X, self.w) + self.b)


# Sample training data (X: Features, y: Labels)
X_train = np.array([[2, 3], [3, 5], [5, 8], [7, 9], [8, 10]])
y_train = np.array([1, 1, -1, -1, -1])  # Class labels (-1, 1)

# Train SVM model
svm = SVM()
svm.fit(X_train, y_train)

# Test data
X_test = np.array([[4, 6], [6, 9]])
predictions = svm.predict(X_test)

print("Predictions:", predictions)  # Output: Predicted class labels


from sklearn.svm import SVC
import numpy as np

"""
Support Vector Machine (SVM) Implementation using Scikit-Learn

📌 **Mathematical Formulation:**
- The SVM algorithm finds the **optimal hyperplane** that maximizes the margin.
- The optimization problem is:

  Minimize:
  𝐿(𝑤) = 1/2 ||w||² + C Σ ξᵢ

  Subject to:
  yᵢ (w · xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

- **Kernel Trick (For Non-Linear SVMs)**:
  If the data is not linearly separable, we use a **kernel function**:

  - **Linear Kernel:** K(xᵢ, xⱼ) = xᵢ · xⱼ
  - **Polynomial Kernel:** K(xᵢ, xⱼ) = (xᵢ · xⱼ + 1)ᵈ
  - **Radial Basis Function (RBF) Kernel:** K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)
"""

# Sample training data
X_train = np.array([[2, 3], [3, 5], [5, 8], [7, 9], [8, 10]])
y_train = np.array([1, 1, -1, -1, -1])  # Class labels (-1, 1)

# Train SVM classifier
svm = SVC(kernel="linear", C=1.0)  # Using a Linear Kernel
svm.fit(X_train, y_train)

# Test data
X_test = np.array([[4, 6], [6, 9]])
predictions = svm.predict(X_test)

print("Predictions:", predictions)


