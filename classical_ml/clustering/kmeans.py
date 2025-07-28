import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, K):
    """
    Randomly selects K initial centroids from the dataset.

    Mathematically:
        μ₁, μ₂, ..., μₖ ← Random(X)

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        K (int): Number of clusters.

    Returns:
        numpy array: Randomly selected K centroids of shape (K, n).
    """
    np.random.seed(42)
    return X[np.random.choice(X.shape[0], K, replace=False)]
def assign_clusters(X, centroids):
    """
    Assigns each data point to the nearest centroid.

    Mathematically:
        C_i = argmin(||X_j - μ_i||²), for all clusters i

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        centroids (numpy array): Current centroids of shape (K, n).

    Returns:
        numpy array: Cluster assignments for each data point.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, K):
    """
    Updates the centroids by computing the mean of each cluster.

    Mathematically:
        μ_i = (1/|C_i|) Σ X_j, for X_j in cluster C_i

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        labels (numpy array): Cluster assignments of shape (m,).
        K (int): Number of clusters.

    Returns:
        numpy array: Updated centroids of shape (K, n).
    """
    return np.array([X[labels == k].mean(axis=0) for k in range(K)])

def k_means(X, K, max_iters=100, tol=1e-4):
    """
    Implements the K-Means clustering algorithm.

    Steps:
    1. Initialize K centroids randomly.
    2. Assign points to the nearest centroid.
    3. Update centroids as the mean of assigned points.
    4. Repeat until convergence or max iterations.

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        tuple: (Final centroids, Cluster assignments)
    """
    centroids = initialize_centroids(X, K)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, K)

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return centroids, labels

np.random.seed(42)
X = np.random.rand(300, 2) 

K = 3
centroids, labels = k_means(X, K)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors="k")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.title("K-Means Clustering (From Scratch)")
plt.legend()
plt.show()


# using scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=3000, centers=3, cluster_std=0.6, random_state=42)

kemans = KMeans(n_clusters=3, random_state=42)
kemans.fit(X)

# Get cluster center and lables
centroids = kemans.cluster_centers_
labels = kemans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors="k")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.title("K-Means Clustering (Using Scikit-Learn)")
plt.legend()
plt.show()













