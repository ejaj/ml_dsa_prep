import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, K, max_iters=100, tol=1e-4):
    """
    K-Means clustering using NumPy (beginner-friendly version with full for-loop).

    Parameters:
        X : np.ndarray of shape (n_samples, n_features)
        K : int, number of clusters
        max_iters : int, maximum number of iterations
        tol : float, tolerance for convergence

    Returns:
        centroids : np.ndarray of shape (K, n_features)
        labels : np.ndarray of shape (n_samples,)
    """
    n_samples, n_features = X.shape

    # Step 1: Randomly initialize centroids from the data
    np.random.seed(42)
    indices = np.random.choice(n_samples, K, replace=False)
    centroids = X[indices]

    for iteration in range(max_iters):
        # Step 2: Compute distances and assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # (n_samples, K)
        labels = np.argmin(distances, axis=1)  # (n_samples,)

        # Step 3: Update centroids using a for-loop
        new_centroids = []
        for k in range(K):
            # Select all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Compute mean of points in cluster k
                new_center = cluster_points.mean(axis=0)
            else:
                # If no points in cluster k, keep the old centroid
                new_center = centroids[k]
            
            new_centroids.append(new_center)
        
        # Convert list to array
        new_centroids = np.array(new_centroids)

        # Step 4: Check for convergence
        diff = np.linalg.norm(new_centroids - centroids)
        print(f"Iteration {iteration}, centroid shift: {diff:.4f}")
        if diff < tol:
            print("Converged.")
            break

        centroids = new_centroids

    return centroids, labels

from sklearn.datasets import make_blobs

# Generate example 2D data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)

# Run K-Means with K=3
final_centroids, final_labels = kmeans(X, K=3)

# Plotting the result
plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', s=30)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', s=200, marker='X')
plt.title("K-Means Clustering (NumPy)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_         # Cluster assignments
centroids = kmeans.cluster_centers_  # Final centroids

# Step 4: Plot the result
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
plt.title("K-Means Clustering (scikit-learn)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()