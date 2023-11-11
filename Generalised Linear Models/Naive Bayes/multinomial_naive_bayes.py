from collections import defaultdict
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_probabilities = defaultdict(float)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))

    def fit(self, X_train, y_train):
        # Calculate class probabilities
        class_counts = np.bincount(y_train)
        self.class_probabilities = class_counts / len(y_train) # This is the prior probability of each class, P©.

        # Calculate feature probabilities for each class
        for label in range(len(class_counts)):
            feature_given_label = X_train[y_train==label]
            self.feature_probabilities[label] = (np.sum(feature_given_label, axis=0) + 1) / (np.sum(feature_given_label) + len(class_counts))
            # The for loop calculates the conditional probability of each feature given each class, P(F|C)
            # Laplace smoothing

    def predict(self, X_test):
        predictions = []
        for instance in X_test:
            posteriors = []
            for label, class_probability in enumerate(self.class_probabilities): # The inner for loop calculates the posterior probability of each class given the features of the instance, P(C|F).
                #  Also, since probabilities can be very small numbers, we use the log function to prevent underflow and to turn the products into sums, which are easier to work with.
                posterior = np.log(class_probability)
                posterior += np.sum(np.log(self.feature_probabilities[label]) * instance)
                posteriors.append(posterior)
                # It starts with the log of the prior probability of the class, and then adds the sum of the logs of the conditional probabilities of the features given the class. The result is the log of the posterior probability
            predictions.append(np.argmax(posteriors))
        return predictions


"""
The Multinomial Naive Bayes algorithm is designed for discrete features (like word counts for text classification). It might not perform well on continuous data, which seems to be your case. If your features are continuous, you might want to consider using Gaussian Naive Bayes instead.

"""


# Sample data (input features and corresponding target values)
# Each feature vector represents the word counts of 'happy', 'sad', 'angry', 'calm' in a text document
X_train = [
    [3, 1, 0, 2],  # Document 1
    [1, 3, 2, 0],  # Document 2
    [0, 2, 3, 1],  # Document 3
    [2, 0, 1, 3]   # Document 4
]

new_test_set =  [
    [2, 1, 0, 2],  # Document 5
    [1, 2, 2, 0],  # Document 6
    [3, 0, 1, 2],  # Document 7
    [0, 3, 2, 1],  # Document 8
    [2, 1, 3, 0],  # Document 9
    [1, 2, 0, 3],  # Document 10
    [3, 1, 2, 0],  # Document 11
    [0, 3, 1, 2],  # Document 12
    [2, 0, 3, 1],  # Document 13
    [1, 2, 3, 0]   # Document 14
]



# The target values represent the classes of the documents
y_train = [0, 1, 2, 0]  # 0: Positive, 1: Negative, 2: Angry

# Test data
X_test = [
    [2, 1, 0, 2],  # Document 5
    [1, 2, 2, 0]   # Document 6
]

X_test.extend(new_test_set)


print("Features:",X_train)

print("Target values:",y_train) 

nvb = MultinomialNaiveBayes()

y_train = np.array(y_train, dtype=int)

nvb.fit(np.array(X_train),y_train) 


# input_data = list(range(1,100,2))
# input_data.extend([1,2,3,14,7,8,16,32,112])
# # input_data = random.sample(range(1, 50), 49)
# correct_output = [0 if i<12 else 1 if i<30 else 2 for i in input_data]
# input_data_scaled = [(i-np.mean(x))/np.std(x) for i in input_data]  Naive Bayes is not distance-based and can handle different ranges for different features.

# Correct output for the test data
y_test_true = [0, 1]
y_test_true.extend([0, 1, 0, 1, 2, 0, 1, 2, 0, 1])

predictions = nvb.predict(X_test)

print("Class probabilities:",nvb.class_probabilities)
print("Feature probabilities:",nvb.feature_probabilities)



print("Test data:", X_test)
print("Correct output:" , y_test_true)
print("Predictions:",predictions)

# correct_predictions = [1 if predictions[i] == correct_output[i] else 0 for i in range(len(correct_output))]

# # Calculate accuracy
# accuracy = sum(correct_predictions) / len(correct_predictions) *100

# print("Accuracy for Naive Bayes:",accuracy,"%")

# Calculate accuracy
correct_predictions = [1 if predictions[i] == y_test_true[i] else 0 for i in range(len(y_test_true))]
accuracy = sum(correct_predictions) / len(correct_predictions) * 100
print("Accuracy:", accuracy, "%")
