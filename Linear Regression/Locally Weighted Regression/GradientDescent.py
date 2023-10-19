import math

def MSE(Y, predictions):
    return (1/len(Y)) * sum(((predictions[i] - Y[i])**2) for i in range(len(Y)))


def locally_weighted_gradient_descent(theta0, theta1, query_point, features , target_values , bandwidth, learning_rate): # technically it is still batch gradient descent
    weights = []
    m = len(target_values)  # number of training examples
    for i in range(m):
        diff = features[i] - query_point
        weights.append(math.exp(diff**2 / (-2 * bandwidth**2))) 

    gradient_theta0 = sum(weights[i]*(theta0 + theta1 * features[i] - target_values[i]) for i in range(m))
    gradient_theta1 = sum(weights[i]*(theta0 + theta1 * features[i] - target_values[i]) * features[i] for i in range(m))

    theta0 = theta0 - learning_rate * gradient_theta0 
    theta1 = theta1 - learning_rate * gradient_theta1

    return theta0,theta1 



# Sample data (input features and corresponding target values)
x = list(range(1,11)) # Input features ; x[0] = 1 (dummy)
# y = random.sample(range(1,16),15)
y = list(i**2.7 + 37*i - i**1.7 + (i**1.732/i**1.44) for i in range(1,11))# Target values
# y = list(i**3.14 for i in range(1,16))# Target values -> 
# y = list(i**12.3456789 for i in range(1,16))# Target values ->
# y = list(i/9 for i in range(1,16))# Target values 
# y = list(i+17 for i in range(1,16))# Target values

print("Features:",x)

print("Target values:",y)


# Initial parameters
theta0_lwr = 0.0 
theta1_lwr = 0.0 

# Learning rate
learning_rate = 0.00001

# Number of iterations
num_iterations = 100000

# error tolerance 
tolerance = 1e-20

bandwidth = 100# important to keep the query point in check , if it strays far from the training set the prediction approaches zero

query_point = 5

prev_cost = float('inf')


for i in range(num_iterations):
    theta0_lwr, theta1_lwr = locally_weighted_gradient_descent(theta0_lwr, theta1_lwr,query_point, x, y, bandwidth, learning_rate)

    predictions = [theta0_lwr + theta1_lwr*i for i in x]

    cost = MSE(y,predictions)

    if abs(cost - prev_cost) < tolerance:
        print(f"Converged at {i} (for Locally weighted)")
        break 
    prev_cost = cost

print(f"Optimized Theta0 for query_point {query_point}:", theta0_lwr)
print(f"Optimized Theta1 for query_point {query_point}:", theta1_lwr)

prediction = theta0_lwr + theta1_lwr*query_point

i = query_point
correct_output = i**2.7 + 37*i - i**1.7 + (i**1.732/i**1.44)

print("Correct Output:",correct_output)
print(f"Prediction for {query_point}:", prediction) 
print("Accuracy:", 100 - abs(correct_output-prediction)/correct_output * 100, "%") 
