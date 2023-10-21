import math
import random

epsilon = 1e-7

"""
The “math domain error” in the cross-entropy function usually occurs when you try to calculate the logarithm of zero or a negative number, which is undefined.
In your case, this can happen if any of the predicted probabilities in your logistic regression model are exactly 0 or 1.
In the logistic regression model, the predicted probabilities are calculated as 1 / (1 + exp(-z)), where z is the linear combination of features and parameters. 
If z is a large positive number, exp(-z) will be close to zero, and the predicted probability will be close to 1. 
If z is a large negative number, exp(-z) will be a large number, and the predicted probability will be close to 0.
When you increase the size of training examples, the model parameters may be updated in such a way that z becomes a large positive or negative number for some examples, leading to predicted probabilities of 0 or 1.
To avoid this issue, you can add a small constant to the predicted probabilities when calculating the logarithm in the cross-entropy function

"""

def cross_entropy(Y, predictions):
    return   sum((Y[i]*math.log(predictions[i] + epsilon ) + (1-Y[i])*math.log(1-predictions[i] + epsilon )) for i in range(len(Y)))


def logistic_gradient_ascent(theta0, theta1, features , target_values , learning_rate): # technically it is still batch gradient descent
    logistic_features = [(1/(1+math.exp(-(theta0 + theta1*i)))) for i in features]
    m = len(target_values)  # number of training examples

    gradient_theta0 = sum((target_values[i] - logistic_features[i]) for i in range(m))
    gradient_theta1 = sum((target_values[i] - logistic_features[i])* features[i] for i in range(m))

    theta0 = theta0 + learning_rate * gradient_theta0 
    theta1 = theta1 + learning_rate * gradient_theta1

    return theta0,theta1 



# Sample data (input features and corresponding target values)
x = list(range(12,50,2)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>30 else 0 for i in x ]
# y = [1 if i%2==0 else 0 for i in x]

"""
Using an odd/even training set would result in an unstable and incorrect model. 
Logistic regression aims to find the best line that fits the training data by minimizing the difference between the predicted probabilities and the actual outcomes. 
However, in the case of odd/even classification, the alternating nature of the outputs makes it impossible for the model to learn accurately

"""

print("Features:",x)

print("Target values:",y)


# Initial parameters
theta0_log = 0.0 
theta1_log = 0.0 

# Learning rate
learning_rate = 0.001

# Number of iterations
num_iterations = 10000

# error tolerance 
tolerance = 1e-6


prev_cost = float('inf')


for i in range(num_iterations):
    theta0_log, theta1_log = logistic_gradient_ascent(theta0_log, theta1_log, x, y, learning_rate)

    predictions = [(1/(1+math.exp(-(theta0_log + theta1_log*i)))) for i in x]

    cost = cross_entropy(y,predictions)

    if abs(cost - prev_cost) < tolerance:
        print(f"Converged at {i} (for Logistic Regression)")
        break 
    prev_cost = cost

print(f"Optimized Theta0 for Logistic Regression:", theta0_log)
print(f"Optimized Theta1 for Logistic Regression:", theta1_log)

input_data = list(range(5,45))
correct_output = [1 if i>30 else 0 for i in input_data ]
# correct_output = [1 if i%2==0 else 0 for i in input_data]


predictions = [1 / (1 + math.exp(-(theta0_log + theta1_log*i))) for i in input_data]

print("Input:", input_data)
print("Correct output:" , correct_output)
print("Prediction:", predictions)

# Convert probabilities to class labels
predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

# Compare predicted and actual class labels
correct_predictions = [1 if predicted_labels[i] == correct_output[i] else 0 for i in range(len(correct_output))]

# Calculate accuracy
accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Logistic Regression:",accuracy,"%")




