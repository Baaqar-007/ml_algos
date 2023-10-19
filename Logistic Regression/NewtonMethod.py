import numpy as np
import math 

epsilon = 1e-7

def sigmoid(x):
    x = float(x)
    if -x > np.log(np.finfo(type(x)).max): # check for overflow
        return 0.0
    else:
        return 1 / (1 + np.exp(-x))

    #return .5 * (1 + np.tanh(.5 * x)) # handling larger inputs

def cross_entropy(Y, predictions):
    return   sum((Y[i]*math.log(predictions[i] + epsilon ) + (1-Y[i])*math.log(1-predictions[i] + epsilon )) for i in range(len(Y)))

def logistic_newtons_method_numpy(X, y, num_iters=10):

    m, n = X.shape
    theta = np.zeros(n) # array of n zeroes

    for _ in range(num_iters):

        h = sigmoid(X @ theta) # calculates the predicted values X multiplied with theta; "@" stands for matrix multiplication

        gradient = X.T @ (h - y) 

        """
        Gradient of the log-likelihood: The gradient of the log-likelihood with respect to the parameters theta is calculated using the formula:
        ∇J(θ) = XT (hθ(X) − y)

        X is the matrix of feature values,
        y is the vector of target values,
        hθ(X) is the vector of predicted probabilities.

        """

        hessian = X.T @ np.diag(h * (1 - h)) @ X # np.diag() creates a diagonal matrix where the diagonal elements are given by h * (1 - h).

        """
        The Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function. In this case, it’s calculated using the formula:

        H = XT.D.X
        where:

        D is a diagonal matrix where the diagonal elements are given by hθ(X)(1−hθ(X)).

        """

        theta = theta - np.linalg.inv(hessian) @ gradient

    return theta


def logistic_newtons_method(theta0, theta1, features , target_values , i):
    m = len(target_values)  # number of training examples

    logistic_features = [sigmoid(theta0 + theta1*features[i]) for i in range(m)]

    gradient_theta0 = sum((target_values[i] - logistic_features[i]) for i in range(m))
    gradient_theta1 = sum((target_values[i] - logistic_features[i]) * features[i] for i in range(m))

    hessian_theta0 = sum((logistic_features[i] * (1 - logistic_features[i])) for i in range(m))
    hessian_theta1 = sum((logistic_features[i] * (1 - logistic_features[i])) * (features[i]**2) for i in range(m))

    theta0 = theta0 - gradient_theta0 / (hessian_theta0 + epsilon)
    theta1 = theta1 - gradient_theta1 / (hessian_theta1 + epsilon)

    return theta0,theta1

# Sample data (input features and corresponding target values)
x = list(range(1,20)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>5 else 0 for i in x ]

print("Features:",x)

print("Target values:",y)


# Initial parameters
theta0_log = 0.5
theta1_log = 0.5

# Learning rate

# Number of iterations
num_iterations = 10000000

# error tolerance 
tolerance = 1e-2


prev_cost = float('inf')

for i in range(num_iterations):
    theta0_log, theta1_log = logistic_newtons_method(theta0_log, theta1_log, x, y, i)

    predictions = [sigmoid(theta0_log + theta1_log*i)for i in x]

    cost = cross_entropy(y,predictions)

    if abs(cost - prev_cost) < tolerance:
        print(f"Converged at {i} (for Logistic Regression)")
        break 
    prev_cost = cost

print(f"Optimized Theta0 for Logistic Regression (Newton's Method):", theta0_log)
print(f"Optimized Theta1 for Logistic Regression (Newton's Method):", theta1_log)

input_data = list(range(1,20,3))
correct_output = [1 if i>5 else 0 for i in input_data ]
# correct_output = [1 if i%2==0 else 0 for i in input_data]


predictions = [sigmoid(theta0_log + theta1_log*i) for i in input_data]

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

