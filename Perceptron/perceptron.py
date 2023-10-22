def step_activation(x):
    if x>=0:
        return 1 
    return 0

# since the step function is not differentiable , Gradient Descent and Newton's Method are not applicable , though they might work for some cases.

def perceptron_updation_rule(theta0, theta1, features , target_values , learning_rate): 

    m = len(target_values)  # number of training examples

    for i in range(m):
        prediction = step_activation(theta0_per + theta1_per*features[i])
        error = target_values[i] - prediction 
        theta0 = theta0 + learning_rate * error 
        theta1 = theta1 + learning_rate * error * features[i]

    return theta0,theta1 



# Sample data (input features and corresponding target values)
x = list(range(1,30,2)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>15 else 0 for i in x ]

print("Features:",x)

print("Target values:",y)


# Initial parameters
theta0_per = 0.0 
theta1_per = 0.0 

# Learning rate
learning_rate = 0.001

# Number of iterations
num_iterations = 10000

# error tolerance 
tolerance = 1e-7


for i in range(num_iterations):
    theta0_per, theta1_per = perceptron_updation_rule(theta0_per, theta1_per, x, y, learning_rate)

print(f"Optimized Theta0 for Perceptron:", theta0_per)
print(f"Optimized Theta1 for Perceptron:", theta1_per)

input_data = list(range(5,45))
input_data.extend([1,2,3,4,5])
correct_output = [1 if i>15 else 0 for i in input_data ]

predictions = [step_activation(theta0_per + theta1_per*i) for i in input_data]

print("Input:", input_data)
print("Correct output:" , correct_output)
print("Prediction:", predictions)

# no need for class labels since the prediction are strictly binary

correct_predictions = [1 if predictions[i] == correct_output[i] else 0 for i in range(len(correct_output))]

# Calculate accuracy
accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Perceptron:",accuracy,"%")




