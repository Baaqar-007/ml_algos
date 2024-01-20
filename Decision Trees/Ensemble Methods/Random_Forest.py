import random
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Create a bootstrap sample
            indices = random.choices(range(len(X)), k=len(X))
            X_sample, y_sample = X[indices], y[indices]

            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier()
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from each tree
        predictions = [tree.predict(X) for tree in self.trees]

        # Return the most common prediction for each instance
        return [max(set(preds), key=preds.count) for preds in zip(*predictions)]
