import numpy as np
import scipy.stats as scp

class GDA:
    def __init__(self):
        self.phi = 0
        self.mu0 = 0
        self.mu1 = 0
        self.sigma = 0

    def fit(self, X, y):
        X = np.array(X).reshape(-1,1)

        m, n = X.shape

        # Compute phi
        self.phi = 1/m * np.sum(y)

        # Compute mu0 and mu1
        self.mu0 = np.sum((1 - y) * X.T, axis=1) / np.sum(1 - y)
        self.mu1 = np.sum(y * X.T, axis=1) / np.sum(y)

        # Compute sigma
        self.sigma = (1/m) * ((X - self.mu0).T @ (X - self.mu0) + (X - self.mu1).T @ (X - self.mu1))

    def predict(self, X):
        p_y_equals_0 = (1 - self.phi) * scp.multivariate_normal.pdf(X, mean=self.mu0, cov=self.sigma)
        p_y_equals_1 = self.phi * scp.multivariate_normal.pdf(X, mean=self.mu1, cov=self.sigma) # the multivariate equation can also be implemented by basic python; the issue is the determinant and the inverse functions they are absolutely not trivial !!
        """
        def multivariate_normal(x, mean, cov):
            n = len(x)
            diff = subtract_vectors(x, mean)
            
            inv_cov = invert_matrix(cov)
            det_cov = determinant(cov)
            
            exponent = -0.5 * dot_product(matrix_multiply([diff], inv_cov)[0], diff)
            
            return math.exp(exponent) / math.sqrt((2*math.pi)**n * det_cov)

        """
        return p_y_equals_1 > p_y_equals_0 

# Sample data (input features and corresponding target values)
x = list(range(12,50,2)) # Input features ; x[0] = 1 (dummy)
y = [1 if i>30 else 0 for i in x ]

print("Features:",x)

print("Target values:",y) 

gda = GDA()

gda.fit(np.array(x),np.array(y)) 

print("mu0:",gda.mu0)
print("mu1:",gda.mu1)
print("sigma:",gda.sigma)
print("phi:",gda.phi)

input_data = list(range(5,45))
input_data.extend([1,2,3,4,5])

correct_output = [1 if i>30 else 0 for i in input_data ] 

predictions = gda.predict(np.array(input_data))

print("Predictions:",predictions)

correct_predictions = [1 if predictions[i] == correct_output[i] else 0 for i in range(len(correct_output))]

# Calculate accuracy
accuracy = sum(correct_predictions) / len(correct_predictions) *100

print("Accuracy for Gaussian Discriminative Analysis:",accuracy,"%")
