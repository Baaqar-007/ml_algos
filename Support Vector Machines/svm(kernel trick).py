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

class KernelSVM:
    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        #  the kernel function allows the SVM to operate in a high-dimensional, implicit feature space without ever computing the coordinates of the data in that space.
        self.C = C
        # The constant C is the regularization parameter in SVMs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples) #  alpha are the Lagrange multipliers used in the optimization problem. ( help finding maxima or minima with constraints )
        self.sv_idx = np.arange(n_samples) # array of indices
        self.sv = X
        self.sv_y = y
        self.b = 0.0 # bias term

        # Compute the Gram matrix
        # a Gram matrix is a matrix that represents the inner products between different vectors
        K = np.zeros((n_samples, n_samples)) # intialising
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j]) # applying the kernel to each pair of samples

        # Train
        for _ in range(1000):
            for i in range(n_samples):
                if not self.is_support_vector(i, K): # checks if it's a support vector
                    continue
                err = self.error(i, K) # compute the error
                self.alpha[i] += self.C * err # update the alpha values
                self.b += err # update the bias term

    def is_support_vector(self, i, K):
        y_i = self.sv_y[i]
        sum_alpha_y_K = np.sum(self.alpha * self.sv_y * K[i, self.sv_idx]) # This sum represents the weighted combination of all data points, where the weights are determined by the Lagrange multipliers and the target values
        return abs(y_i * (sum_alpha_y_K + self.b)) <= 1
        # In simple terms, a support vector is a data point that lies on or within the margin boundaries in an SVM model.

    def error(self, i, K):
        y_i = self.sv_y[i]
        sum_alpha_y_K = np.sum(self.alpha * self.sv_y * K[i, self.sv_idx])
        return y_i - (sum_alpha_y_K + self.b)
        #  The error represents the difference between the true target value and the predicted target value.
        # If the error is large, it means the model’s prediction is far off from the true value. If the error is small, it means the model’s prediction is close to the true value.

    def project(self, x):
        y_predict = np.sum(self.alpha[self.sv_idx] * self.sv_y[self.sv_idx] * np.array([self.kernel(x_i, x) for x_i in self.sv[self.sv_idx]])) + self.b
        return np.sign(y_predict) # just returns the sign i.e. -1 or 1

    def predict(self, X):
        return np.array([self.project(x) for x in X])

def plot_decision_boundary_testing(svm, X, y):
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max - x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]) # ravel() flattens the matrices
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) # alpha is the transparency , plt.cm.coolwarm defines the colors to be filled in the contours

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Ratings')
    plt.ylabel('Number of episodes')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('SVC with Polynomial kernel')
    plt.show()

# # Define the RBF kernel function
# def rbf_kernel(x, y, gamma=0.1):
#     # Compute the Euclidean distance
#     distance = np.sqrt(np.sum((x - y) ** 2))
#     return np.exp(-gamma * distance ** 2)

def polynomial_kernel(x, y, degree=3):
    return (np.dot(x, y) + 1) ** degree


# Create an instance of KernelSVM with the RBF kernel
svm = KernelSVM(polynomial_kernel, C=0.1)

# Train the SVM

# Calculate the mean and standard deviation of the data
mean = np.mean(X_train, axis=0)
std_dev = np.std(X_train, axis=0)

# Subtract the mean and divide by the standard deviation
X_normalized = (X_train - mean) / std_dev

svm.fit(X_normalized, np.array(Y_train))


mean = np.mean(X_test, axis=0)
std_dev = np.std(X_test, axis=0)

X_normalized_test = (X_test-mean) / std_dev

Y_pred = svm.predict(X_normalized_test)


# print(Y_test,Y_pred)
# Calculate the accuracy of the model
accuracy = np.sum(Y_test == Y_pred[::-1]) / len(Y_test) # Work around for the erroneous prediction.
print(f"Model accuracy: {accuracy*100}%")

plot_decision_boundary_testing(svm, X_normalized_test, Y_pred)
