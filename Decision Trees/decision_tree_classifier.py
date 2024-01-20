from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import pair_confusion_matrix
import seaborn as sns
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\baaqa\\OneDrive\\Documents\\Machine Learning\\Decision Trees\\Ensemble Methods')
from Random_Forest import RandomForest 
from AdaBoost import AdaBoost


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class # prediction
        self.feature_index = 0 # feature value
        self.threshold = 0 # threshold value, to decide to which side to go to
        self.left = None 
        self.right = None

class DecisionTreeClassifier:
    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y) # grows the decision tree using features X, labels y

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs): # moves along the tree comparing at each node.
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _grow_tree(self, X, y):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class) # most common class
        node = Node(predicted_class=predicted_class)

        if len(set(y)) == 1: # reached leaf node
            return node

        feature_ix, threshold = self._best_split(X, y) # finds the best split between the right and left
        if feature_ix is not None:
            indices_left = X[:, feature_ix] < threshold # splits the node into left and right based on the threshold value
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.feature_index = feature_ix
            node.threshold = threshold
            node.left = self._grow_tree(X_left, y_left)
            node.right = self._grow_tree(X_right, y_right)
        return node

    def _best_split(self, X, y):
        """
        The Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
        """

        # the _best_split method implements a greedy algorithm that tries all possible splits and selects the one that results in the smallest Gini impurity. 

        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes)] # number of samples in each class
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent) # for parent node
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y))) # sorts the samples by the feature values
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m): # iterates over each value of threshold
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                """ Transfers one sample from right to left child"""
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m # weighted average for the both the childs
                if thresholds[i] == thresholds[i - 1]: # if same threshold as the previous one , skips the iteration
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

iris = load_iris()
X = iris.data
y = iris.target
n_classes = len(set(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_trees = 100 
n_learners = 100
# Initialize and fit the models
model_dt = DecisionTreeClassifier()
model_rf = RandomForest(n_trees=n_trees)  
model_ab = AdaBoost(n_learners=n_learners)  

model_dt.fit(X_train, y_train, n_classes)
model_rf.fit(X_train, y_train)  
model_ab.fit(X_train, y_train)  

# Predict the test data
y_pred_dt = model_dt.predict(X_test)
y_pred_rf = model_rf.predict(X_test)  
y_pred_ab = model_ab.predict(X_test)  

# Compute the confusion matrices
cm_dt = pair_confusion_matrix(y_test, y_pred_dt)
cm_rf = pair_confusion_matrix(y_test, y_pred_rf)
cm_ab = pair_confusion_matrix(y_test, y_pred_ab)

# Create a figure with 3 subplots
fig, axs = plt.subplots(ncols=3, figsize=(30,10))

# Plot the confusion matrices
sns.heatmap(cm_dt, annot=True, ax=axs[0])
sns.heatmap(cm_rf, annot=True, ax=axs[1])
sns.heatmap(cm_ab, annot=True, ax=axs[2])

# Set titles
axs[0].set_title('Decision Tree')
axs[1].set_title('Random Forest')
axs[2].set_title('AdaBoost')

# Set labels
for ax in axs:
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')

plt.show()

"""

S.NO        Bagging                                                                                            Boosting

1. The simplest way of combining predictions that belong to the same type.    A way of combining predictions that belong to the different types.

2.  Aim to decrease variance, not bias.                                       Aim to decrease bias, not variance.

3.  Each model receives equal weight.                                         Models are weighted according to their performance.

4.  Each model is built independently.                                        New models are influenced by the performance of previously built models.

5.  Different training data subsets are selected using row sampling with      Every new subset contains the elements that were misclassified by previous models.
 replacementand random sampling methods from the entire training dataset.    

6.  Bagging tries to solve the over-fitting problem.                          Boosting tries to reduce bias.

7.  If the classifier is unstable (high variance), then apply bagging.        If the classifier is stable and simple (high bias) the apply boosting.

8.  In this base classifiers are trained parallelly.                          In this base classifiers are trained sequentially.

9   Example: The Random forest model uses Bagging.                            Example: The AdaBoost uses Boosting techniques

"""



"""
Bagging is like asking each friend to guess the fruit independently, and then we choose the fruit that most friends guessed. Each friend makes their best guess, and in the end, we trust the wisdom of the crowd.

Boosting, on the other hand, is like asking our friends to guess one by one. The first friend makes a guess. Then the second friend tries to correct the mistakes of the first friend. The third friend tries to correct the mistakes of the first two friends, and so on. In the end, we combine all their guesses, giving more importance to the guesses of friends who corrected more mistakes.

"""