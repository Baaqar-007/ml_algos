# one step to find global minimum 
import numpy as np
import random

def get_theta(features, target_values): 

	# Ensure features is a 2D array (a column vector)
	features = np.array(features).reshape(-1, 1)
	# Ensure target_values is a 2D array (a column vector)
	target_values = np.array(target_values).reshape(-1, 1)

	#When adding a column to represent the bias term in linear regression, it's common to use ones for a specific reason related to the nature of the bias term. 
	#The bias term represents the value of the dependent variable when all independent variables are zero. 
	#By setting the added column to ones, you're effectively saying that when all the features are zero, the model predicts the output to be equal to the bias term.

	features_with_bias = np.hstack((np.ones_like(features), features)) # adding a column of ones

	transposed = np.transpose(features_with_bias)

	A = np.dot(transposed, features_with_bias) 

	B = np.linalg.inv(A) 

	theta = np.dot(np.dot(B,transposed),target_values)

	return theta


x = list(range(1,16)) # Input features ; x[0] = 1 (dummy)
# y = [4, 5, 3, 12, 7, 15, 6, 1, 8, 13, 10, 14, 11, 9, 2]
# y = random.sample(range(1,16),15)
# y = list(int(i**2.7 + 37*i - i**1.7 + (i**1.732//i**1.44)) for i in range(1,16)) # Target values
y = list(i**12.3456789 for i in range(1,16))
# y = list(i+17 for i in range(1,16))


print("Features:",x)

print("Target values:",y)

print("Features after reshaping: ",np.array(x).reshape(-1, 1))

print("Features after adding a column of ones for the bias term: ",np.hstack((np.ones_like(x), x)) )

theta = get_theta(x,y)

print("Optimised theta0 by normal equation: ",theta[0])
print("Optimised theta1 by normal equation: ",theta[1])

input_data = [16,17,18,19]
predictions_neq= []
for i in input_data:
    predictions_neq.append(theta[0] + theta[1]*i)

# correct_output = list(int(i**2.7 + 37*i - i**1.7 + (i**1.732//i**1.44)) for i in range(16,20))
# correct_output = list(i + 17 for i in range(16,20)) 
correct_output = list(i**12.3456789 for i in range(16,20))

print("Input:", input_data)
print("Correct output:" , correct_output)
print("Prediction by Normal equation:", predictions_neq)

error = [] 

for i in range(len(input_data)):
	error.append((abs(correct_output[i]-predictions_neq[i][0])/correct_output[i]) * 100)


#print("Error:", sum(error[0])/len(error), "%")
print("Accuracy for normal equation :",100-sum(error)/len(error),"%")