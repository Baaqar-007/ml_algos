import numpy as np
import math 

epsilon = 1e-7

def sigmoid(x):
    # if -x > np.log(np.finfo(type(x)).max): # check for overflow
    #     return 0.0
    # else:
    return 1 / (1 + np.exp(-x))

    #return .5 * (1 + np.tanh(.5 * x)) # handling larger inputs

def cross_entropy(Y, predictions):
    return   sum((Y[i]*math.log(predictions[i] + epsilon ) + (1-Y[i])*math.log(1-predictions[i] + epsilon )) for i in range(len(Y)))

def logistic_newtons_method_numpy(X, y, num_iters=10):

    X = np.array(X).reshape(-1,1)

    X = np.hstack((np.ones_like(X), X)) # bias term

    m, n = X.shape # returns a tuple of sizes of its dimensions

    theta = np.zeros(n) # array of n zeroes

    for _ in range(num_iters):

        h = sigmoid(X @ theta) # calculates the predicted values X multiplied with theta; "@" stands for matrix multiplication

        gradient = X.T @ (y - h) 

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
        hessian += np.eye(hessian.shape[0]) * 1e-5 
        # In this code, np.eye(hessian.shape[0])  creates an identity matrix with the same size as the Hessian, and * 1e-5 scales it by a small constant. 
        # When this ridge is added to the Hessian, it becomes invertible even if it was originally singular
        # shape[0], returns the size of the first dimension of the array

        theta = theta + np.linalg.inv(hessian) @ gradient

    return theta


# Sample data (input features and corresponding target values)
x = list(range(2,20)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>5 else 0 for i in x ]

print("Features:",x)

print("Target values:",y)

# # Initial parameters
# theta0_log = 0.5
# theta1_log = 0.5

# Number of iterations
num_iterations = 10000

# error tolerance 
# tolerance = 1e-2

theta_log = logistic_newtons_method_numpy(x, y, num_iterations)

query_point = 1

print(f"Prediction for {query_point}:", sigmoid(theta_log[0] + theta_log[1]*query_point))


# prev_cost = float('inf')

# for i in range(num_iterations):
#     theta0_log, theta1_log = logistic_newtons_method(theta0_log, theta1_log, x, y, i)

#     predictions = [sigmoid(theta0_log + theta1_log*i)for i in x]

#     cost = cross_entropy(y,predictions)

#     if abs(cost - prev_cost) < tolerance:
#         print(f"Converged at {i} (for Logistic Regression)")
#         break 
#     prev_cost = cost

print(f"Optimized Theta0 for Logistic Regression (Newton's Method):", theta_log[0])
print(f"Optimized Theta1 for Logistic Regression (Newton's Method):", theta_log[1])

input_data = list(range(1,30))
input_data.extend([1,2,3,4,5])
correct_output = [1 if i>5 else 0 for i in input_data ]


predictions = [sigmoid(theta_log[0] + theta_log[1]*i) for i in input_data]

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

