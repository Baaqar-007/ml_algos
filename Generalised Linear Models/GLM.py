import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exp_link(x):
    return np.exp(x)

def identity_link(x):
    return x

# Generalized Linear Models (GLMs) are a flexible framework for modeling a response variable's relationship with one or more predictor variables,
# where the response variable follows a specified probability distribution.

def glm(X, y, link='identity', num_iters=100000):
    # Initialize theta
    X = np.array(X).reshape(-1,1)

    y = np.array(y)

    X = np.hstack((np.ones_like(X), X)) # for bias term

    theta = np.zeros(X.shape[1])

    # Choose the link function
    if link == 'logit':
        link_function = sigmoid
    elif link == 'log':
        link_function = exp_link
    else:  # 'identity'
        link_function = identity_link

    # In the context of machine learning algorithms, we divide by m (the number of samples) to calculate the average cost over all samples. 
    # This is known as the mean cost or average loss. 
    # By calculating the average, we ensure that the total cost is not influenced by the size of the dataset.

    for i in range(num_iters):
        h = link_function(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= 0.0001 * gradient

    return theta


# Sample data (input features and corresponding target values)
X = list(range(1,100)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>57 else 0 for i in X ]

print("Features:",X)

print("Target values:",y)

# Linear Regression
theta_linear = glm(X, y, link='identity')

# Logistic Regression
theta_logistic = glm(X, y, link='logit')

# Poisson Regression
theta_poisson = glm(X, y, link='log')
print("-"*80)
print(f"Optimized Theta0 for Linear Regression (Newton's Method):", theta_linear[0])
print(f"Optimized Theta1 for Linear Regression (Newton's Method):", theta_linear[1])

print("-"*80)

print(f"Optimized Theta0 for Logistic Regression (Newton's Method):", theta_logistic[0])
print(f"Optimized Theta1 for Logistic Regression (Newton's Method):", theta_logistic[1])

print("-"*80)

print(f"Optimized Theta0 for Poisson Regression (Newton's Method):", theta_poisson[0])
print(f"Optimized Theta1 for Poisson Regression (Newton's Method):", theta_poisson[1])

print("-"*80)

input_data = random.sample(range(150),40)
correct_output = [1 if i>57 else 0 for i in input_data ]

prediction_linear = [identity_link(theta_linear[0] + theta_linear[1]*i) for i in input_data]
prediction_logistic = [sigmoid(theta_logistic[0] + theta_logistic[1]*i) for i in input_data]
prediction_poisson = [exp_link(theta_poisson[0] + theta_poisson[1]*i) for i in input_data]
print("Correct output:" , correct_output)



predicted_labels_linear = [1 if p >= 0.5 else 0 for p in prediction_linear]
predicted_labels_logistic = [1 if p >= 0.5 else 0 for p in prediction_logistic]
predicted_labels_poisson = [1 if p >= 0.5 else 0 for p in prediction_poisson]

print("-"*80)
print("Prediction for Linear Regression:", predicted_labels_linear)
print("-"*80)
print("Prediction for Logistic Regression:", predicted_labels_logistic)
print("-"*80)
print("Prediction for Poisson Regression:", predicted_labels_poisson)
print("-"*80) 

correct_predictions_linear = [1 if predicted_labels_linear[i] == correct_output[i] else 0 for i in range(len(correct_output))]
correct_predictions_logistic = [1 if predicted_labels_logistic[i] == correct_output[i] else 0 for i in range(len(correct_output))]
correct_predictions_poisson = [1 if predicted_labels_poisson[i] == correct_output[i] else 0 for i in range(len(correct_output))]


accuracy_linear = sum(correct_predictions_linear) / len(correct_predictions_linear) *100
accuracy_logistic = sum(correct_predictions_logistic) / len(correct_predictions_logistic) *100
accuracy_poisson = sum(correct_predictions_poisson) / len(correct_predictions_poisson) *100


print("Accuracy for Linear Regression:",accuracy_linear,"%")
print("Accuracy for Logistic Regression:",accuracy_logistic,"%")
print("Accuracy for Poisson Regression:",accuracy_poisson,"%")


