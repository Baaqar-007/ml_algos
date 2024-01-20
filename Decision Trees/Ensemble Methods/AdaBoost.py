import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_learners):
        self.n_learners = n_learners
        self.learners = []
        self.alphas = [] # stores individual weights

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

"""
Algorithm:

1) Initialise the dataset and assign equal weight to each of the data point.
2) Provide this as input to the model and identify the wrongly classified data points.
3) Increase the weight of the wrongly classified data points and decrease the weights of correctly classified data points. And then normalize the weights of all data points.
4) if (got required results)
  Goto step 5
else
  Goto step 2
5) End
"""
