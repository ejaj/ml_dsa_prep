import numpy as np
from collections import Counter

class KNN:

    def __init__(self, K=3):
        self.K = K  # Number of neighbors
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
    def euclidean_distance(self, x1, x2):
        """Compute Euclidean Distance"""      
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def predict(self, X_test):
        """Predict labels for test points"""
        predictions = []
        for x in X_test:
            distances = []

            # Compute distance to all training points
            for i, x_train in enumerate(self.X_train):
                dist = self.euclidean_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and select the K nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_neighbors = [label for _, label in distances[:self.K]]
            # Majority vote for classification
            most_common = Counter(k_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

X_train = np.array([[2, 4], [4, 6], [4, 8], [6, 9], [2, 5]])
y_train = np.array(["Dog", "Dog", "Cat", "Cat", "Dog"])

X_test = np.array([[3, 7], [5, 8]])  # Test points

# Train KNN and make predictions
knn = KNN(K=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print("Predictions:", predictions)

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample dataset
X_train = np.array([[2, 4], [4, 6], [4, 8], [6, 9], [2, 5]])
y_train = np.array(["Dog", "Dog", "Cat", "Cat", "Dog"])

X_test = np.array([[3, 7], [5, 8]])  # Test points

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Set K=3
knn.fit(X_train, y_train)

# Predict test points
predictions = knn.predict(X_test)
print("Predictions:", predictions)
