import math
import numpy as np

# By Normal equation
def locally_weighted_regression(query_point, features , target_values , bandwidth):
    m = len(target_values)  # number of training examples
    n = len(features)# number of features
    features = np.array(features).reshape(-1, 1)
    # Add a column of ones for theta0
    X = np.hstack((np.ones((m, 1)), features))  # for bias/intercept term

    y = np.array(target_values).reshape(-1, 1)

    query_point = np.hstack(([1], query_point))

    # Calculate weights for each training example
    #The np.eye() function is used in the code to create a square 2-D array with ones on the main diagonal and zeros elsewhere. 
    #This is also known as an identity matrix.
    weights = np.eye(m)
    #In the context of Locally Weighted Regression (LWR), this identity matrix is used as a starting point to create a weight matrix. 
    #The weight matrix is a diagonal matrix where each entry along the diagonal, Wii, corresponds to the weight for the i-th training example. 
    #These weights are computed based on the distance of each training example from the query point, with closer examples receiving higher weights.

    for i in range(m):
        diff = X[i] - query_point
        weights[i, i] = np.exp(diff.dot(diff.T) / (-2 * bandwidth**2))

    # Solve normal equation "pinv" calculates pseudo-inverse
    theta = np.linalg.pinv(X.T.dot(weights).dot(X)).dot(X.T).dot(weights).dot(y)
    # np.linalg.pinv() is preferred over np.linalg.inv() in linear regression due to its ability to handle non-square and singular matrices and provide more stable numerical solutions3.
    # Make prediction
    prediction = query_point.dot(theta)

    return prediction



x = list(range(1,16)) 
y = list(i**2 for i in range(1,16))

print("Features:",x)
print("Traget values:",y)

bandwidth = 0.5
query_point = 16

predicted_value = locally_weighted_regression(query_point,x,y,bandwidth)

print("Predicted value at x =", query_point, "is", predicted_value) 

input_data = list(range(20,30))
predictions_lwr= []
for i in input_data:
    predictions_lwr.append(locally_weighted_regression(i,x,y,bandwidth)[0])

# correct_output = list(int(i**2.7 + 37*i - i**1.7 + (i**1.732//i**1.44)) for i in range(16,20))
# correct_output = list(i + 17 for i in range(16,20)) 
correct_output = list(i**2 for i in range(20,30))

print("Input:", input_data)
print("Correct output:" , correct_output)
print("Prediction by Locally weighted regression:", predictions_lwr)

error = [] 

for i in range(len(input_data)):
    error.append((abs(correct_output[i]-predictions_lwr[i])/correct_output[i]) * 100)


#print("Error:", sum(error[0])/len(error), "%")
print("Accuracy for locally_weighted_regression :",100-sum(error)/len(error),"%")