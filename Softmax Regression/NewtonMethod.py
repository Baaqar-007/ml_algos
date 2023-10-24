import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x)) # subtracting the maximum value in x for numerical stability.
    return e_x / e_x.sum(axis=1, keepdims=True) # normalisation ; The keepdims=True argument ensures that the sum is kept as a 2D array with one column, which allows for correct broadcasting during the division operation.

def newton_method(X, y, num_classes, num_iterations):
    num_samples, num_features = X.shape
    weights = np.zeros((num_features, num_classes))

    for _ in range(num_iterations):
        scores = np.dot(X, weights)
        predictions = softmax(scores)

        # Create one-hot encoded matrix
        one_hot_y = np.zeros_like(predictions) # 49X3 matrix for the given training set
        
        # if _ == 0:
        #     print(one_hot_y)

        # if _ == 0:
        #     print(np.arange(len(y)),y)

        one_hot_y[np.arange(len(y)), y] = 1

        # In one-hot encoding, each label is represented as a binary vector that has a length equal to the number of classes and has a 1 at the index of the true class and 0s elsewhere.

        # if _ == 0:
        #     print(one_hot_y)

        gradient = np.dot(X.T, (predictions - one_hot_y))

        # Calculate the Hessian
        hessian = np.zeros((num_features, num_features))
        for i in range(num_samples):
            for c in range(num_classes):
                hessian += predictions[i, c] * (1 - predictions[i, c]) * np.outer(X[i], X[i])

                """
                np.outer(a,b) computes the outer product of two vectors. If a and b are 1-D arrays, it forms a new 2-D array c such that c[i,j] = a[i] * b[j]. 
                It’s used in this code to calculate each term in the sum for the Hessian matrix.

                """
        hessian += np.eye(hessian.shape[0]) * 1e-5

        weights -= np.dot(np.linalg.inv(hessian), gradient)

    return weights

def predict(X, weights):
    X = np.array(X).reshape(-1, 1)

    # Add bias term to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    scores = np.dot(X, weights)
    predictions = softmax(scores)
    return np.argmax(predictions, axis=1)

# Usage:
# x and y are your original lists
x = list(range(1,50)) # Input features ; x[0] = 1 (dummy)
y = [0 if i<12 else 1 if i<30 else 2 for i in x ]
x_scaled = [(i-np.mean(x))/np.std(x) for i in x]

print("Features:",x)
# print("Scaled Features:",x_scaled)
print("Target values:",y)


# Convert lists to numpy arrays and reshape X to be 2D
X = np.array(x_scaled).reshape(-1, 1)
y = np.array(y)

# Add bias term to X
X = np.hstack((np.ones((X.shape[0], 1)), X))

num_classes = len(set(y)) # number of classes
num_iterations = 1000
weights = newton_method(X, y, num_classes, num_iterations)

print(f"Optimized Theta0 and Theta1 for first class:", weights[0][0], weights[1][0])
print(f"Optimized Theta0 and Theta1 for second class:", weights[0][1], weights[1][1])
print(f"Optimized Theta0 and Theta1 for third class:", weights[0][2], weights[1][2])


input_data = list(range(1,100,2))
input_data.extend([1,2,3,14,7,8,16,32,112])
correct_output = [0 if i<12 else 1 if i<30 else 2 for i in input_data]
input_data_scaled = [(i-np.mean(x))/np.std(x) for i in input_data]

predictions = predict(input_data_scaled,weights)

print("Input:", input_data)
print("Correct output:" , correct_output)
# print("Scaled Input:",input_data_scaled)
print("Prediction:", list(predictions))

correct_predictions = [1 if predictions[i] == correct_output[i] else 0 for i in range(len(correct_output))]

accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Softmax Regression:",accuracy,"%")
