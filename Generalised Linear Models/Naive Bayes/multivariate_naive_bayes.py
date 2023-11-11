import numpy as np
from collections import defaultdict
from scipy.stats import norm
"""
In binary classification, each feature is assumed to be a binary variable. Therefore, the likelihood of a feature given a class can be calculated directly from the proportions in the training data, without needing to estimate any parameters of a distribution.

For example, if you're classifying emails as spam or not spam and one of your features is "contains the word 'free'", then the likelihoods are simply the proportions of spam and not-spam emails that contain the word 'free' in your training data. These proportions can be calculated directly without needing to estimate a mean or standard deviation.

However, when dealing with continuous or multi-valued discrete features, we can't calculate these proportions directly. Instead, we assume some kind of distribution for our data. A common choice is the Gaussian distribution for continuous features, which is defined by two parameters: the mean (average) and standard deviation (measure of dispersion). We estimate these parameters from our training data for each feature and each class. These estimated parameters are then used to calculate the likelihoods needed for prediction.

So, in summary, standard deviations (and means) are not needed when doing binary classification because we can calculate likelihoods directly from proportions in the training data. But they are needed when dealing with continuous or multi-valued discrete features because we need to estimate parameters of a distribution to calculate likelihoods.

"""

class NaiveBayes:
    def __init__(self):
        self.class_probabilities = defaultdict(float) # This will hold the probabilities of each class in our Naive Bayes classifier.
        self.feature_probabilities = defaultdict(lambda: defaultdict(float)) # feature_probabilities is another dictionary that returns a dictionary as a default value which in turn returns float as a default value. This will hold the probabilities of each feature given a class.
        # These probabilities are stored in a nested dictionary where the first key is the feature and the second key is the class. The value is the calculated probability.
        self.feature_stddev = defaultdict(lambda: defaultdict(float))

    def fit(self, X_train, y_train):
        # Calculate class probabilities
        X_train = np.array(X_train).reshape(-1,1)

        class_counts = np.bincount(y_train) # calculates the number of classes
        self.class_probabilities = class_counts / len(y_train) # probability of each class

        # Calculate feature probabilities for each class
        for feature in range(X_train.shape[1]): # each feature
            for label in range(len(class_counts)): # each class
                feature_given_label = X_train[y_train==label, feature] # Selecting all samples of the current feature where the class label is equal to the current class ; boolean mask
                # print(feature_given_label); separates test data into different classes
                self.feature_probabilities[feature][label] = np.mean(feature_given_label) #  This mean value is an estimate of the probability of seeing this feature given this class.
                self.feature_stddev[feature][label] = np.std(feature_given_label)
        # The result is a set of conditional probabilities for each feature given each class. These probabilities are used to make predictions on new data.

    def predict(self, X_test):
        X_test = np.array(X_test).reshape(-1,1)
        predictions = []
        for instance in X_test: #  For each instance, it calculates the posterior probability for each class and then makes a prediction
            posteriors = []
            for label, class_probability in enumerate(self.class_probabilities): # The prior probability of a class is the proportion of instances in the training data that belong to that class.
                posterior = class_probability # The posterior probability is initialized with the prior probability of the class.
                for feature_index, feature_value in enumerate(instance):
                    likelihood = norm.pdf(feature_value, self.feature_probabilities[feature_index][label], self.feature_stddev[feature_index][label])
                    # Gaussian probability density function : This is used to calculate the likelihood of a feature value given a class.
                    posterior *= likelihood # corresponds to multiplying the likelihood and prior; (P(X∣C)P(C))
                    #P(C∣X) is the posterior probability of class C given features X,
					#P(X∣C) is the likelihood which is the probability of features X given class C,
					#P(C) is the prior which is the probability of class,

                posteriors.append(posterior)
            # print(f"Posterior probs for dataset {instance} : {posteriors}")
            predictions.append(np.argmax(posteriors)) # corresponds to finding the class with the highest posterior probability.(index)
        # print("p:",predictions)
        return predictions

""" For Binary classification :
    def predict(self, X_test):
        X_test = np.array(X_test).reshape(-1,1)
        predictions = []
        for instance in X_test:
            posteriors = []
            for label, class_probability in enumerate(self.class_probabilities):
                posterior = class_probability
                for feature_index, feature_value in enumerate(instance):
                    likelihood = self.feature_probabilities[feature_index][label] if feature_value == 1 else 1 - self.feature_probabilities[feature_index][label]
                    posterior *= likelihood
                posteriors.append(posterior)
            predictions.append(np.argmax(posteriors))
        return predictions
"""

# Sample data (input features and corresponding target values)
x = list(range(1,50)) # Input features ; x[0] = 1 (dummy)
y = [0 if i<12 else 1 if i<30 else 2 for i in x ]
x_scaled = [(i-np.mean(x))/np.std(x) for i in x]

print("Features:",x)

print("Target values:",y) 

nvb = NaiveBayes()

nvb.fit(x_scaled,np.array(y)) 


input_data = list(range(1,100,2))
input_data.extend([1,2,3,14,7,8,16,32,112])
# input_data = random.sample(range(1, 50), 49)
correct_output = [0 if i<12 else 1 if i<30 else 2 for i in input_data]
input_data_scaled = [(i-np.mean(x))/np.std(x) for i in input_data]

predictions = nvb.predict(input_data_scaled)

print("Class probabilities:",nvb.class_probabilities)
print("Feature probabilities:",nvb.feature_probabilities)
print("Standard deviations:",nvb.feature_stddev)



print("Test data:", input_data)
print("Correct output:" , correct_output)
print("Predictions:",predictions)

correct_predictions = [1 if predictions[i] == correct_output[i] else 0 for i in range(len(correct_output))]

# Calculate accuracy
accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Naive Bayes:",accuracy,"%")

"""
Sure, let's imagine you're playing a guessing game. You have a box full of different kinds of fruits: apples, oranges, and bananas. Now, each of these fruits has certain features. For example, apples are usually red, oranges are orange, and bananas are yellow and long.

Now, suppose you pick a fruit from the box without looking, and I give you some hints about the fruit. I tell you the fruit is orange in color. You would then guess that the fruit is probably an orange because oranges are the only fruit in the box that are orange.

But what if I tell you the fruit is yellow? It could be an apple or a banana because both can be yellow. But if I give you another hint and say that the fruit is long, you would guess it's a banana because bananas are the only long fruit in the box.

This is kind of what Naive Bayes does. It's like playing a guessing game. It uses hints (which we call features in machine learning) about an item to guess what class (like the type of fruit) the item belongs to. And just like how you guessed the fruit was a banana because it was yellow and long, Naive Bayes combines all the hints to make a guess.

But why is it called 'Naive'? Well, it's because it makes a 'naive' assumption that all hints are independent of each other, which means knowing one hint doesn't change the chances of another hint being true. This is like assuming that knowing a fruit is yellow doesn't change the chances of it being long. This isn't always true in real life (like how bananas are more likely to be long if they're yellow), but this 'naive' assumption often works well enough and makes the math easier for Naive Bayes.

"""