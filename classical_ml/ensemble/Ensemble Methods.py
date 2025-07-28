import numpy as np

class DecisionStump:
    """
    A simple weak learner that classifies data based on a threshold over a single feature.
    """

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weights=None):
        """
        Trains the stump using the feature and threshold that minimize the (weighted) classification error.

        Args:
            X (np.ndarray): Feature matrix (m x n)
            y (np.ndarray): Labels (+1 or -1)
            sample_weights (np.ndarray): Weights for each sample
        """
        m, n = X.shape
        if sample_weights is None:
            sample_weights = np.ones(m) / m

        min_error = float('inf')

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(m)
                    pred[polarity * X[:, feature] < polarity * threshold] = -1
                    error = np.sum(sample_weights[pred != y])
                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature

    def predict(self, X):
        """
        Predict using the trained feature, threshold, and polarity.

        Returns:
            np.ndarray: Array of +1 or -1 predictions
        """
        pred = np.ones(X.shape[0])
        feature_vals = X[:, self.feature_index]
        pred[self.polarity * feature_vals < self.polarity * self.threshold] = -1
        return pred

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([-1, -1, +1, +1, +1])

stump = DecisionStump()
stump.fit(X, y)

print("Threshold:", stump.threshold)
print("Feature index:", stump.feature_index)
print("Polarity:", stump.polarity)
print("Prediction:", stump.predict(X))

class AdaBoost:
    """
    AdaBoost ensemble using DecisionStumps as weak learners.
    """

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        m = X.shape[0]
        w = np.ones(m) / m  # initialize weights

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, w)
            pred = stump.predict(X)

            error = np.sum(w[pred != y])
            error = max(error, 1e-10)  # avoid divide-by-zero

            alpha = 0.5 * np.log((1 - error) / error)

            w *= np.exp(-alpha * y * pred)
            w /= np.sum(w)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            final_pred += alpha * model.predict(X)
        return np.sign(final_pred)

class VotingEnsemble:
    """
    Hard or Soft voting ensemble.
    """

    def __init__(self, models, voting='hard', weights=None):
        self.models = models
        self.voting = voting
        self.weights = weights

    def predict(self, X):
        all_preds = np.array([model.predict(X) for model in self.models])
        
        if self.voting == 'hard':
            # Majority vote (±1)
            return np.sign(np.sum(all_preds, axis=0))

        elif self.voting == 'soft':
            # Weighted vote (assuming confidence is represented by model weights)
            if self.weights is None:
                self.weights = np.ones(len(self.models))
            weighted_preds = np.tensordot(self.weights, all_preds, axes=(0, 0))
            return np.sign(weighted_preds)

        else:
            raise ValueError("Voting must be 'hard' or 'soft'")
