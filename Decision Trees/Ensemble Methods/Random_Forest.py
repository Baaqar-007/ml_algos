import random
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = [] # stores individual decision trees

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
        predictions = [tree.predict(X) for tree in self.trees] # prediction from each tree
        # Return the most common prediction for each instance
        return [max(set(preds), key=preds.count) for preds in zip(*predictions)]
        # zip(*predictions) transposes the list of predictions so that each inner list contains the predictions for a single instance from all trees.

"""
Implementation Steps of Bagging

Step 1: Multiple subsets are created from the original data set with equal tuples, selecting observations with replacement.
Step 2: A base model is created on each of these subsets.
Step 3: Each model is learned in parallel with each training set and independent of each other.
Step 4: The final predictions are determined by combining the predictions from all the models.

"""
