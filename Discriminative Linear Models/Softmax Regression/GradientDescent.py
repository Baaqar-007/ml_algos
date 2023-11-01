import math
import numpy as np

def softmax(z):
    e_z = [math.exp(i) for i in z]
    sum_e_z = sum(e_z)
    return [i / sum_e_z for i in e_z]

def cross_entropy(y, y_hat):
    return -sum(y[i] * math.log(y_hat[i]) for i in range(len(y)))

def softmax_regression(X, y, num_iters=1000, learning_rate=0.001, tolerance = 1e-5):

    m = len(X)
    K = len(set(y))  # Number of classes
    flag = 0

    theta = [[0.0]*2 for _ in range(K)]  # Initialize theta

    for _ in range(num_iters):

        gradient = [[0.0]*2 for _ in range(K)]

        for i in range(m):
            x = X[i]

            scores = [sum(x*theta[k][j] for j in range(2)) for k in range(K)]

            probabilities = softmax(scores) # normalisation ; gives a probability distribution

            y_encoded = [1 if y[i] == k else 0 for k in range(K)]

            loss = cross_entropy(y_encoded, probabilities) / m

            # Break if the loss is less than the tolerance
            if loss < tolerance:
                flag = 1

            for k in range(K):
                for j in range(2):
                    if j == 0 :
                        gradient[k][j] += (probabilities[k] - y_encoded[k])
                    else:
                        gradient[k][j] += (probabilities[k] - y_encoded[k])*x
        if flag == 1 :
            print(f"Converged at {_} (for softmax regression)")
            break

        theta = [[theta[k][j] - learning_rate * gradient[k][j] / m for j in range(2)] for k in range(K)]

    return theta

def predict(x, theta):
    K = len(theta)  # Number of classes
    scores = [x*theta[k][1] + theta[k][0] for k in range(K)]
    probabilities = softmax(scores)  # Gives a probability distribution
    return probabilities.index(max(probabilities))  # Returns the class with the highest probability


x = list(range(1,50)) # Input features ; x[0] = 1 (dummy)
y = [0 if i<12 else 1 if i<30 else 2 for i in x ]
x_scaled = [(i-np.mean(x))//np.std(x) for i in x]

print("Features:",x)
print("Scaled Features:",x_scaled)
print("Target values:",y)

theta = softmax_regression(x_scaled,y) # for two parameters theta0 and theta1

print(f"Optimized Theta0 and Theta1 for first class:", theta[0])
print(f"Optimized Theta0 and Theta1 for second class:", theta[1])
print(f"Optimized Theta0 and Theta1 for third class:", theta[2])

input_data = list(range(1,100,2))
input_data.extend([1,2,3,4,7,8,6,32,112])
correct_output = [0 if i<12 else 1 if i<30 else 2 for i in input_data]
input_data_scaled = [(i-np.mean(x))//np.std(x) for i in input_data]

predictions = [predict(i, theta) for i in input_data_scaled]

print("Input:", input_data)
print("Correct output:" , correct_output)
print("Scaled Input:",input_data_scaled)
print("Prediction:", predictions)

correct_predictions = [1 if predictions[i] == correct_output[i] else 0 for i in range(len(correct_output))]

accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Softmax Regression:",accuracy,"%")
