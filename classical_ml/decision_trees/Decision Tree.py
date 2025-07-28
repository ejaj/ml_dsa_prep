from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import pandas as pd

data = pd.DataFrame({
    "color": ["Red", "Red", "Green", "Red", "Green"],
    "label": ["Apple", "Apple", "Pear", "Apple", "Pear"]
})

data_encoded = pd.get_dummies(data[["color"]])

print(data_encoded)

labels = data["label"]

clf = DecisionTreeClassifier(criterion="gini")
clf.fit(data_encoded, labels)

tree.plot_tree(clf, feature_names=data_encoded.columns, class_names=clf.classes_, filled=True)


test = pd.DataFrame({"color": ["Red", "Green"]})
test_encoded = pd.get_dummies(test)
predictions = clf.predict(test_encoded)

print("Predictions:", predictions)

y_true = ["Apple", "Pear"]  
y_pred = predictions 

acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))