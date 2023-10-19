
import numpy as np
import math 

epsilon = 1e-7

def sigmoid(x):
    if -x > np.log(np.finfo(type(x)).max): # check for overflow
        return 0.0
    else:
        return 1 / (1 + np.exp(-x))

def cross_entropy(Y, predictions):
    return -sum((Y[i]*math.log(predictions[i] + epsilon ) + (1-Y[i])*math.log(1-predictions[i] + epsilon )) for i in range(len(Y)))

def logistic_newtons_method(theta0, theta1, features , target_values , num_iterations=10000, tolerance=1e-7):
    m = len(target_values)  # number of training examples

    prev_cost = float('inf')

    for _ in range(num_iterations):
        logistic_features = [sigmoid(theta0 + theta1*features[i]) for i in range(m)]

        gradient_theta0 = sum((target_values[i] - logistic_features[i]) for i in range(m))
        gradient_theta1 = sum((target_values[i] - logistic_features[i]) * features[i] for i in range(m))

        hessian_theta0 = sum((logistic_features[i] * (1 - logistic_features[i])) for i in range(m))
        hessian_theta1 = sum((logistic_features[i] * (1 - logistic_features[i])) * (features[i]**2) for i in range(m))

        theta0 = theta0 -  gradient_theta0 / (hessian_theta0 + epsilon)
        theta1 = theta1 -  gradient_theta1 / (hessian_theta1 + epsilon)

        predictions = [sigmoid(theta0 + theta1*i)for i in features]

        cost = cross_entropy(target_values,predictions)

        if abs(cost - prev_cost) < tolerance:
            print(f"Converged at iteration {_}")
            break 
        prev_cost = cost

    return theta0,theta1

# Sample data (input features and corresponding target values)
x = list(range(2,20)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>15 else 0 for i in x ]

# Scale features
x_scaled = [(i-np.mean(x))/np.std(x) for i in x]

print("Features:",x_scaled)
print("Target values:",y)

# Initial parameters
theta0_log = 0
theta1_log = 0


theta0_log, theta1_log = logistic_newtons_method(theta0_log, theta1_log, x_scaled, y)

query_point = 8

print(f"Prediction for {query_point}:", sigmoid(theta0_log + theta1_log*(query_point-np.mean(x))/np.std(x)))

print(f"Optimized Theta0 for Logistic Regression (Newton's Method):", theta0_log)
print(f"Optimized Theta1 for Logistic Regression (Newton's Method):", theta1_log)

input_data = list(range(1,20))
# input_data.extend([1,2,3,4,5])
# input_data.extend([112,334,56])
# input_data.extend([11,12,14])
correct_output = [1 if i>15 else 0 for i in input_data ]

input_data_scaled = [(i-np.mean(x))/np.std(x) for i in input_data]

predictions = [sigmoid(theta0_log + theta1_log*i) for i in input_data_scaled]

print("Input:", input_data)
print("Scaled Input:",input_data_scaled)
print("Correct output:" , correct_output)
print("Prediction:", predictions)

# Convert probabilities to class labels
predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

print("Prediction Labels:", predicted_labels)

# Compare predicted and actual class labels
correct_predictions = [1 if predicted_labels[i] == correct_output[i] else 0 for i in range(len(correct_output))]

# Calculate accuracy
accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Logistic Regression:",accuracy,"%")
