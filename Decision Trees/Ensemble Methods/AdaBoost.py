import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_learners):
        self.n_learners = n_learners
        self.learners = []
        self.alphas = []

    def fit(self, X, y):
        weights = np.full(len(X), 1 / len(X))

        for _ in range(self.n_learners):
            # Train a decision stump (a decision tree with max_depth=1)
            learner = DecisionTreeClassifier(max_depth=1)
            learner.fit(X, y, sample_weight=weights)
            self.learners.append(learner)

            # Calculate the error rate
            predictions = learner.predict(X)
            error_rate = np.sum(weights * (predictions != y)) / np.sum(weights)

            # Calculate the learner's weight in the ensemble
            alpha = 0.5 * np.log((1 - error_rate) / error_rate)
            self.alphas.append(alpha)

            # Update the instance weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        # Get weighted predictions from each learner
        predictions = [alpha * learner.predict(X) for alpha, learner in zip(self.alphas, self.learners)]

        # Return the sign of the sum of predictions for each instance
        return np.sign(np.sum(predictions, axis=0))
