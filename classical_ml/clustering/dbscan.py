import numpy as np
from collections import deque
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
class DBSCAN:
    def __init__(self, eps=0.5, minPts=5, distance_metric='euclidean'):
        """
        Density-Based Clustering Algorithm (DBSCAN)

        Parameters:
        eps (float): Radius for density neighborhood (ε).
        minPts (int): Minimum points required to form a dense region.
        distance_metric (str): Distance metric ('euclidean' or 'mahalanobis').
        """
        self.eps = eps
        self.minPts = minPts
        self.distance_metric = distance_metric
        self.labels_ = None
    def fit(self, X):
        """
        Applies DBSCAN clustering on dataset X.

        Parameters:
        X (numpy array): Dataset of shape (m, n).

        Returns:
        self.labels_ (numpy array): Cluster labels (-1 for noise).
        """
        n = X.shape[0]
        self.labels_ = np.full(n, -1)  # -1 means unclassified (noise)
        cluster_id = 0
        
        for i in range(n):
            if self.labels_[i] != -1: # Already visited
                continue
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.minPts:
                self.labels_[i] = -1 # Mark as noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1 # Move to next cluster
        return self.labels_
    

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """
        Expands the cluster using density reachability.

        Parameters:
        X (numpy array): Dataset.
        point_idx (int): Current core point.
        neighbors (list): List of neighboring points.
        cluster_id (int): Current cluster ID.
        """
        self.labels_[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()
            if self.labels_[neighbor_idx] == -1:  # Previously noise, make it a border point
                self.labels_[neighbor_idx] = cluster_id
            
            if self.labels_[neighbor_idx] != -1:  # Already assigned
                continue

            self.labels_[neighbor_idx] = cluster_id
            new_neighbors = self._region_query(X, neighbor_idx)

            if len(new_neighbors) >= self.minPts:
                queue.extend(new_neighbors)  # Expand cluster further
    def _region_query(self, X, point_idx):
        """
        Finds all points within ε-neighborhood of a point.

        Parameters:
        X (numpy array): Dataset.
        point_idx (int): Index of point.

        Returns:
        List of neighbor indices.
        """
        neighbors = []
        for i in range(X.shape[0]):
            if self._compute_distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _compute_distance(self, p, q):
        """
        Computes distance between two points based on selected metric.
        """
        if self.distance_metric == 'euclidean':
            return euclidean(p, q)
        elif self.distance_metric == 'mahalanobis':
            inv_cov_matrix = np.linalg.inv(np.cov(X.T))
            return np.sqrt((p - q).T @ inv_cov_matrix @ (p - q))
        else:
            raise ValueError("Invalid distance metric!")

X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, minPts=5)
labels = dbscan.fit(X)

# Plot Results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors="k")
plt.title("DBSCAN Clustering from Scratch")
plt.show()


from sklearn.cluster import DBSCAN

# Generate dataset (moon-shaped clusters)
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)


# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5).fit(X)
dbscan_labels = dbscan.labels_

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ax[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', edgecolors="k")
ax[1].set_title("DBSCAN Clustering")

plt.show()

import math

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def region_query(data, point_idx, eps):
    neighbors = []
    for idx, point in enumerate(data):
        if euclidean_distance(data[point_idx], point) <= eps:
            neighbors.append(idx)
    return neighbors

def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_pts):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        
        if labels[neighbor_idx] == -1:  # Previously marked as noise
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:  # Unvisited
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(data, neighbor_idx, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors
        i += 1

def dbscan(data, eps, min_pts):
    labels = [0] * len(data)  # 0 = unvisited, -1 = noise, >0 = cluster ID
    cluster_id = 0

    for point_idx in range(len(data)):
        if labels[point_idx] != 0:
            continue  # Already visited

        neighbors = region_query(data, point_idx, eps)

        if len(neighbors) < min_pts:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_pts)

    return labels

data = [
    [1, 2], [2, 2], [2, 3],
    [8, 7], [8, 8], [25, 80]
]

# Run DBSCAN
labels = dbscan(data, eps=2, min_pts=2)

print("Cluster labels:", labels)

