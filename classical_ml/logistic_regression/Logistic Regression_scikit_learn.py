import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_data():
    """
    Generate a synthetic binary classification dataset.

    Mathematically:
        - X ∈ ℝ^(m×n): Feature matrix with `m` samples and `n` features.
        - y ∈ {0,1}: Binary labels.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    X, y = make_classification(n_samples=500, n_features=2, random_state=42)
    return X, y


def normalize_data(X_train, X_test):
    """
    Standardize features using mean and standard deviation.

    Mathematically:
        X_norm = (X - μ) / σ

    where:
        - μ: mean of each feature.
        - σ: standard deviation of each feature.

    Parameters:
        X_train (numpy array): Training data.
        X_test (numpy array): Testing data.

    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model using scikit-learn.

    Mathematically:
        h_w(X) = σ(Xw) = 1 / (1 + exp(-Xw))

    Parameters:
        X_train (numpy array): Training feature matrix.
        y_train (numpy array): Training labels.

    Returns:
        model: Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using accuracy score.

    Mathematically:
        Accuracy = (Correct Predictions / Total Samples) * 100

    Parameters:
        model: Trained logistic regression model.
        X_test (numpy array): Test feature matrix.
        y_test (numpy array): Test labels.

    Returns:
        float: Accuracy percentage.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)
    return y_pred


def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary of the trained logistic regression model.

    Mathematically:
        Decision boundary satisfies:
        Xw = 0   ->   w0 + w1*x1 + w2*x2 = 0

    Parameters:
        model: Trained logistic regression model.
        X (numpy array): Feature matrix.
        y (numpy array): Target labels.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict on grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title("Decision Boundary (Logistic Regression with scikit-learn)")
    plt.show()


# Step 1: Generate dataset
X, y = generate_data()

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Normalize dataset
X_train, X_test = normalize_data(X_train, X_test)

# Step 4: Train logistic regression model
model = train_logistic_regression(X_train, y_train)

# Step 5: Evaluate the model
y_pred = evaluate_model(model, X_test, y_test)

# Step 6: Plot decision boundary
plot_decision_boundary(model, X, y)
