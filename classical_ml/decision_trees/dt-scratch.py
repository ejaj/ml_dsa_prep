import numpy as np


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log(probs))


def best_split(X, y):
    best_gain = -1
    best_feature, best_threshold = None, None
    base_entropy = entropy(y)

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left = y[X[:, feature] <= t]
            right = y[X[:, feature] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            p_left, p_right = len(left) / len(y), len(right) / len(y)
            gain = base_entropy - (p_left * entropy(left) + p_right * entropy(right))
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t
    return best_feature, best_threshold


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def build_tree(X, y, depth=0, max_depth=0):
    if len(set(y)) == 1 or depth == max_depth:
        return Node(value=np.bincount(y).argmax())

    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=np.bincount(y).argmax())

    indices_left = X[:, feature] <= threshold
    left = build_tree(X[indices_left], y[indices_left], depth + 1, max_depth)
    right = build_tree(X[~indices_left], y[~indices_left], depth + 1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left, right=right)


def predict(tree, sample):
    if tree.value is not None:
        return tree.value
    if sample[tree.feature] <= tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)


from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target
X, y = X[y < 2], y[y < 2]  # binary classification for simplicity

tree = build_tree(X, y)
preds = [predict(tree, sample) for sample in X]
print("Accuracy:", np.mean(preds == y))

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def best_threshold(X, y):
    best_gain = -1
    best_thresh = None
    base_entropy = entropy(y)

    thresholds = np.unique(X)
    for t in thresholds:
        left_mask = X <= t
        right_mask = X > t
        y_left = y[left_mask]
        y_right = y[right_mask]
        if len(y_left) == 0 or len(y_right) == 0:
            continue
        p_left, p_right = len(y_left) / len(y), len(y_right) / len(y)
        entropy_left = entropy(y_left)
        entropy_right = entropy(y_right)
        weighted_entropy = p_left * entropy_left + p_right * entropy_right
        info_gain = base_entropy - weighted_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_thresh = t
    return best_thresh, best_gain


X = np.array([2.5, 3.5, 6.0, 7.5])
y = np.array([0, 0, 1, 1])

threshold, gain = best_threshold(X, y)
print("Best Threshold:", threshold)
print("Information Gain:", gain)
