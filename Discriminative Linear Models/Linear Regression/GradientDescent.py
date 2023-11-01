
import random

def MSE(Y, predictions):
    return 0.5 * sum(((predictions[i] - Y[i])**2) for i in range(len(Y)))


def batch_gradient_descent(theta0, theta1, features , target_values , learning_rate):

    m = len(target_values)  # number of training examples

    gradient_theta0 = 0.5 * sum((theta0 + theta1 * features[i] - target_values[i]) for i in range(m))
    gradient_theta1 = 0.5 * sum((theta0 + theta1 * features[i] - target_values[i]) * features[i] for i in range(m))

    theta0 = theta0 - learning_rate * gradient_theta0 
    theta1 = theta1 - learning_rate * gradient_theta1

    return theta0,theta1

def stochastic_gradient_descent(theta0, theta1, features , target_values , learning_rate, i):

    gradient_theta0 = 0.5 * (theta0 + theta1 * features[i] - target_values[i])
    gradient_theta1 = 0.5 * (theta0 + theta1 * features[i] - target_values[i]) * features[i] 

    theta0 = theta0 - learning_rate * gradient_theta0 
    theta1 = theta1 - learning_rate * gradient_theta1


    return theta0,theta1



# Sample data (input features and corresponding target values)
x = list(range(1,16)) # Input features ; x[0] = 1 (dummy)
# y = random.sample(range(1,16),15)
y = list(int(i**2.7 + 37*i - i**1.7 + (i**1.732//i**1.44)) for i in range(1,16))# Target values -> around 70% accuracy
# y = list(i**3.14 for i in range(1,16))# Target values -> 55% accuracy
# y = list(i**12.3456789 for i in range(1,16))# Target values -> 58% 
# y = list(i/9 for i in range(1,16))# Target values -> 99.8%
# y = list(i+17 for i in range(1,16))# Target values

print("Features:",x)

print("Target values:",y)

# Initial parameters
theta0_batch = 0.0 
theta1_batch = 0.0

theta0_stochastic = 0.0 
theta1_stochastic = 0.0

# Learning rate
learning_rate = 0.000001

# Number of iterations
num_iterations = 100000

# error tolerance 
tolerance = 1e-15

prev_cost = float('inf')
#Gradient Descent
for i in range(num_iterations):
    theta0_batch, theta1_batch = batch_gradient_descent(theta0_batch, theta1_batch, x, y, learning_rate)

    predictions = [theta0_batch + theta1_batch*i for i in x]

    cost = MSE(y,predictions)

    if abs(cost - prev_cost) < tolerance:
        print(f"Converged at {i} (for Batch)")
        break 
    prev_cost = cost

prev_cost = float('inf')

for i in range(num_iterations):
    m = len(x) # number of training examples
    flag = 0
    for j in range(m):
        theta0_stochastic, theta1_stochastic = stochastic_gradient_descent(theta0_stochastic, theta1_stochastic, x, y, learning_rate,j)
        predictions = [theta0_stochastic + theta1_stochastic * i for i in x]

        cost = MSE(y,predictions)

        if abs(cost - prev_cost) < tolerance:
            flag = 1 
            break 
        prev_cost = cost

        if i%1000:
            learning_rate *= 0.999991

    if flag == 1 :
        print(f"Converged at {i} (for Stochastic)")
        break

# Print optimized parameters
print("Optimized Theta0 for Batch Descent:", theta0_batch)
print("Optimized Theta1 for Batch Descent:", theta1_batch)

print("Optimized Theta0 for Stochastic Descent:", theta0_stochastic)
print("Optimized Theta1 for Stochastic Descent:", theta1_stochastic)


input_data = [16,17,18,19]
predictions_batch = []
predictions_stochastic = []
for i in input_data:
    predictions_batch.append(theta0_batch + theta1_batch*i)

for i in input_data:
    predictions_stochastic.append(theta0_stochastic + theta1_stochastic*i) 

correct_output = list(int(i**2.7 + 37*i - i**1.7 + (i**1.732//i**1.44)) for i in range(16,20))
# correct_output = list(i + 17 for i in range(16,20)) 
# correct_output = list(i**12.3456789 for i in range(16,20))
# correct_output = list(i**3 for i in range(16,20))
# correct_output = list(i/9 for i in range(16,20))

print("Input:", input_data)
print("Correct output:" , correct_output)
print("Prediction by Batch Descent:", predictions_batch)
print("Prediction by Stochastic Descent:", predictions_stochastic)

error_batch = [] 
error_stochastic = [] 

for i in range(len(input_data)):
    error_batch.append(abs(correct_output[i]-predictions_batch[i])/correct_output[i] * 100) 

for i in range(len(input_data)):
    error_stochastic.append(abs(correct_output[i]-predictions_stochastic[i])/correct_output[i] * 100) 


print("Accuracy for Batch Descent:", 100-sum(error_batch)/len(error_batch), "%")
print("Accuracy for Stochastic Descent:", 100-sum(error_stochastic)/len(error_stochastic), "%")