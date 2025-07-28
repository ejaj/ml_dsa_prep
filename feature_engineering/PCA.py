import numpy as np
import matplotlib.pyplot as plt

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components  # Number of principal components to keep
        self.mean = None
        self.components = None
    
    def fit(self, X):
        """Compute PCA components from input dataset X"""
        # Step 1: Standardize the data (Subtract mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean  # Center the data

        # Step 2: Compute the Covariance Matrix
        cov_matrix = np.cov(X_centered.T)  # Covariance of features

        # Step 3: Compute Eigenvalues & Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

         # Step 4: Sort Eigenvectors by Eigenvalues (Descending Order)
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Get indices of sorted eigenvalues
        eigenvectors = eigenvectors[:, sorted_indices]  # Sort eigenvectors
        eigenvalues = eigenvalues[sorted_indices]  # Sort eigenvalues


        # Step 5: Select Top K Eigenvectors (Principal Components)
        self.components = eigenvectors[:, :self.n_components]
    def transform(self, X):
        """Project data onto the new PCA space"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


# Generate sample 3D dataset
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features

# Apply PCA
pca = PCAFromScratch(n_components=2)  # Reduce from 3D to 2D
pca.fit(X)
X_reduced = pca.transform(X)

# Plot Reduced Data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color='blue', alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA (Dimensionality Reduction from 3D to 2D)")
plt.show()


from sklearn.decomposition import PCA
# Generate Sample Data (3D)
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
# Apply PCA
pca = PCA(n_components=2)  # Reduce from 3D to 2D
X_reduced = pca.fit_transform(X)

# Explained Variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot Reduced Data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color='red', alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA (Scikit-Learn Implementation)")
plt.show()

