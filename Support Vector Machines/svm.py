import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define our data
np.random.seed(0)
# Generate the dataset
X1 = np.random.randn(20, 2) - [2, 2]  # first class
X2 = np.random.randn(20, 2) + [2, 2]  # second class
X = np.r_[X1, X2]

Y1 = [-1] * 20  # labels for first class
Y2 = [1] * 20   # labels for second class
Y = np.r_[Y1, Y2]


# X is a 40x2 array where the first half of the rows are random points roughly centered around (-2, -2), and the second half are random points roughly centered around (2, 2).


# Define the size of the test set
test_size = 0.5

# Calculate the number of examples in the test set
test_size = int(len(Y) * test_size)

# Split the dataset into a training set and a testing set
X_train = np.r_[X1[:-test_size//2], X2[:-test_size//2]]
X_test = np.r_[X1[-test_size//2:], X2[-test_size//2:]]
Y_train = np.r_[Y1[:-test_size//2], Y2[:-test_size//2]]
Y_test = np.r_[Y1[-test_size//2:], Y2[-test_size//2:]]


# Step 2: Define the SVM class
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param # soft margin ; regularisation parameter ; the smaller it is ,the more relaxed is functional margin
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):  # Uses Stochastic Descent
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1 # checking the condition i.e if the functional margin is greater than 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                    # Here, lr is the learning rate, lambda is the regularization parameter, x_i is the feature vector of the i-th sample, and y_i is the label of the i-th sample. The dot product dot(x_i, y_i) is computed only when the condition y_i * (dot(x_i, w) - b) < 1 is not met.
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Step 3: Train the SVM
svm = SVM()
svm.fit(X, np.array(Y))

# Step 4: Visualize the results


def plot_decision_boundary_testing(svm, X, y):
    def f(x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y) # color decided by y
    a0 = -4; a1 = f(a0, svm.w, svm.b)
    b0 = 4; b1 = f(b0, svm.w, svm.b)
    plt.plot([a0,b0], [a1,b1], 'k') # decision boundary

    a0 = -4; a1 = f(a0, svm.w, svm.b, 1)
    b0 = 4; b1 = f(b0, svm.w, svm.b, 1)
    plt.plot([a0,b0], [a1,b1], 'k--')

    a0 = -4; a1 = f(a0, svm.w, svm.b, -1)
    b0 = 4; b1 = f(b0, svm.w, svm.b, -1)
    plt.plot([a0,b0], [a1,b1], 'k--')

    # a0 and a1 define the start point of the line. a0 is the x-coordinate and a1 is the y-coordinate
    # b0 and b1 define the end point of the line. b0 is the x-coordinate and b1 is the y-coordinate.

    plt.show()

# Testing the model 


# Predict the labels of the test set
Y_pred = svm.predict(X_test)

# Calculate the accuracy of the model
accuracy = sum(Y_test == Y_pred) / len(Y_test)
print(f"Model accuracy: {accuracy*100}%")


# Plot for Testing data 
plot_decision_boundary_testing(svm, X_test, Y_pred)


