class DecisionTreeClassifier:
    def __init__(self):
        """
        Initializes an empty Decision Tree.
        The 'tree' will store a simple one-level decision based on the best binary split.
        """
        self.tree = None

    def gini_impurity(self, labels):
        """
        Calculates the Gini Impurity for a list of class labels.

        Formula:
            Gini(D) = 1 - Σ (p_i)^2
        Where:
            - D is the dataset (list of labels)
            - p_i is the probability of class i in the dataset D

        Interpretation:
            - Gini = 0 → Pure node (only one class)
            - Gini → Maximum when classes are equally mixed (high uncertainty)
        """
        total = len(labels)
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        impurity = 1
        for label in counts:
            prob = counts[label] / total
            impurity -= prob ** 2
        return impurity

    def split_data(self, data, feature, value):
        """
        Splits the dataset into two subsets based on the binary condition:
            data[feature] == value

        Returns:
            - true_branch: samples where feature == value
            - false_branch: samples where feature != value

        This is used to evaluate how well a given feature-value pair splits the data.
        """
        true_branch, false_branch = [], []
        for row in data:
            if row[feature] == value:
                true_branch.append(row)
            else:
                false_branch.append(row)
        return true_branch, false_branch

    def info_gain(self, true_branch, false_branch, current_uncertainty):
        """
        Calculates the Information Gain of a split.

        Formula:
            IG(D, A) = Gini(D) - [ |D_left| / |D| * Gini(D_left) + |D_right| / |D| * Gini(D_right) ]

        Where:
            - D is the parent dataset before splitting
            - A is the feature being tested
            - D_left and D_right are the resulting subsets
            - Gini(D) is the impurity before the split
            - IG measures reduction in impurity after the split

        Interpretation:
            - Higher Information Gain = better feature for splitting
        """
        p = len(true_branch) / (len(true_branch) + len(false_branch))
        gain = current_uncertainty - (
            p * self.gini_impurity([row["label"] for row in true_branch]) +
            (1 - p) * self.gini_impurity([row["label"] for row in false_branch])
        )
        return gain

    def fit(self, data, feature, value):
        """
        Trains the decision tree using a single binary split.

        This 'fit' function performs a one-level split based on the specified feature and value.
        It:
            1. Calculates the overall Gini impurity of the dataset.
            2. Splits the data based on (feature == value).
            3. Calculates information gain from the split.
            4. Stores the best split and branches in self.tree.
        """
        labels = [row["label"] for row in data]
        current_gini = self.gini_impurity(labels)
        true_branch, false_branch = self.split_data(data, feature, value)
        gain = self.info_gain(true_branch, false_branch, current_gini)

        self.tree = {
            "feature": feature,
            "value": value,
            "gain": gain,
            "true_branch": true_branch,
            "false_branch": false_branch
        }

    def print_tree(self):
        """
        Prints the simple decision tree (1-level).

        Output format:
            If feature == value: → label from true_branch
            Else: → label from false_branch

        Assumes branches are pure for simplicity.
        """
        print("Tree:")
        print(f"If {self.tree['feature']} == {self.tree['value']}: → {self.tree['true_branch'][0]['label']}")
        print(f"Else: → {self.tree['false_branch'][0]['label']}")

data = [
    {"color": "Red", "label": "Apple"},
    {"color": "Red", "label": "Apple"},
    {"color": "Green", "label": "Pear"},
    {"color": "Red", "label": "Apple"},
    {"color": "Green", "label": "Pear"},
]

tree = DecisionTreeClassifier()
tree.fit(data, feature="color", value="Red")
tree.print_tree()
