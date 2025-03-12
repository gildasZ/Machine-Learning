'''
HOMEWORK 05
'''

###
# Define the rtf/txt file path
output_file_path = "HMW_05/ML_HW5_output.txt"

""" Input path for I. Gaussian Process"""
input_file_path_gaussian_process = 'HMW_05/data/input.data'

""" Input paths for II. SVM on MNIST dataset"""
X_train_file_path = "HMW_05/data/X_train.csv"
Y_train_file_path = "HMW_05/data/Y_train.csv"
X_test_file_path = "HMW_05/data/X_test.csv"
Y_test_file_path = "HMW_05/data/Y_test.csv"

# Open output file in write mode and clear its contents (if it exists)
output_file = open(output_file_path, 'w')
output_file.close()  # This will clear the contents if the file already exists

# Reopen the output file in append mode to keep it open
rtf_file = open(output_file_path, 'a')

# Define a custom "pseudo personal print" function
def perprint(message, end = None, file = rtf_file):
    print(message, end = end)
    print(message, end = end, file = file)
###

perprint("Starting...\n")


##
### I. Gaussian Process
###    In this section, you are going to implement Gaussian Process and visualize the result.
perprint("I. Gaussian Process \n")
##

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd

# Training data
input_data = np.loadtxt(input_file_path_gaussian_process)
X = input_data[:, 0].reshape(-1, 1)  # Input features
Y = input_data[:, 1].reshape(-1, 1)  # Observed values
#perprint(X, end ="\n\n")
#perprint(y, end ="\n\n")

# Define the rational quadratic kernel
def Rational_Quadratic_Kernel(X1, X2, sigma, alpha, length_scale):
    """ Using rational quadratic kernel function: k(x_i, x_j) = (sigma^2) * (1 + (x_i-x_j)^2 / (2*alpha * length_scale^2))^-alpha. """
    #square_distance = np.power(X1.reshape(-1,1) - X2.reshape(1,-1), 2)
    square_distance = np.power(ssd.cdist(X1, X2), 2)
    #perprint(square_error, end ="\n\n")
    kernel = (sigma ** 2) * np.power( 1 + square_distance/(2*alpha*length_scale**2), -alpha )
    return kernel

# Define the Gaussian Process Regression
def Gaussian_Process_Regression(X, Y, new_X, beta, sigma, alpha, length_scale):
    """ Gaussian process regression with rational quadratic kernel. """

    K_train_train = Rational_Quadratic_Kernel(X, X, sigma, alpha, length_scale) # From training X
    C = K_train_train + (1 / beta) * np.eye(len(X)) # Covariance matrix

    inv_C = np.linalg.inv(C)
    K_train_new = Rational_Quadratic_Kernel(X, new_X, sigma, alpha, length_scale)
    K_new_new = Rational_Quadratic_Kernel(new_X, new_X, sigma, alpha, length_scale)
    K_new_new += (1 / beta) * np.eye(len(new_X))

    mean_new = np.dot(K_train_new.T, np.dot(inv_C, Y))
    cov_new = K_new_new - np.dot(K_train_new.T, np.dot(inv_C, K_train_new))

    return mean_new, cov_new

# Define the Negative Marginal Log-Likelihood
def Negative_Marginal_Log_Likelihood(params, X, Y, beta):
    sigma, alpha, length_scale = params

    K_train_train = Rational_Quadratic_Kernel(X, X, sigma, alpha, length_scale)
    C = K_train_train + (1 / beta) * np.eye(len(X))
    inv_C = np.linalg.inv(C)

    neg_marg_log_lik = 0.5 * len(X) * np.log(2 * np.pi)
    neg_marg_log_lik += 0.5 * np.sum(np.log(np.diagonal(np.linalg.cholesky(C)))) # More stable
    #neg_marg_log_lik += 0.5 * np.log(np.linalg.det(C))
    neg_marg_log_lik += 0.5 * np.dot(Y.T, np.dot(inv_C, Y))

    return neg_marg_log_lik

# Define the Visualizati
def Visualize_Gaussian_Process(X_train, Y_train, new_X, mean_new, cov_new, sigma, alpha, length_scale):
    plt.figure(figsize=(10, 6))

    # Plot training data points
    plt.scatter(X_train, Y_train, color='k', label='Training Data')

    # Draw a line to represent mean of f in range [-60,60].
    plt.plot(new_X, mean_new, color='b', label='Mean of f')

    interval = 1.96 * np.sqrt(np.diag(cov_new))
    new_X = new_X.ravel()
    mean_new = mean_new.ravel()
    # Mark the 95% confidence interval of f.
    plt.fill_between(new_X, mean_new + interval, mean_new - interval, color='r', alpha=0.3, label='95% Confidence Interval of f')

    plt.legend()
    plt.xlim(-60, 60)
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.title('Gaussian Process Regression with Rational Quadratic Kernel')
    plt.title(f'sigma: {sigma:.5f}, alpha: {alpha:.5f}, length scale: {length_scale:.5f}')
    #plt.grid(True)
    plt.show()

if __name__ == '__main__':
    perprint("   Task 1: \n")
    # Get new points in range [-60,60]
    new_X = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    # Parameter beta
    beta = 5
    # Kernel parameters
    sigma = 1
    alpha = 1
    length_scale = 1

    # Perform Gaussian Process Regression
    mean_new, cov_new = Gaussian_Process_Regression(X, Y, new_X, beta, sigma, alpha, length_scale)
    # Visualize the result
    Visualize_Gaussian_Process(X, Y, new_X, mean_new, cov_new, sigma, alpha, length_scale)


    ###
    perprint("\n   Task 2: \n")
    # Initial kernel parameters
    initial_params = [sigma, alpha, length_scale]

    # Optimize kernel parameters
    optimized = minimize(Negative_Marginal_Log_Likelihood, initial_params,
                        bounds = ((1e-7, None), (1e-7, None), (1e-7, None)),
                        args=(X, Y, beta)).x  # Don't forget .x
    sigma = optimized[0]
    alpha = optimized[1]
    length_scale = optimized[2]

    # Perform Gaussian Process Regression
    mean_new, cov_new = Gaussian_Process_Regression(X, Y, new_X, beta, *optimized)

    # Visualize the result
    Visualize_Gaussian_Process(X, Y, new_X, mean_new, cov_new, sigma, alpha, length_scale)



#pip install libsvm-official

##
### II. SVM on MNIST dataset
###    In this section, you are going to implement Gaussian Process and visualize the result.
perprint("II. SVM on MNIST dataset \n")
##

from libsvm.svmutil import * #svm_read_problem, svm_train, svm_predict


###
perprint("\n   Task 1: \n")

# Load training and testing data
X_train = np.loadtxt(X_train_file_path, delimiter=',')
Y_train = np.loadtxt(Y_train_file_path)
X_test = np.loadtxt(X_test_file_path, delimiter=',')
Y_test = np.loadtxt(Y_test_file_path)

# Convert Y_train and Y_test to integers (if not already)
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# Define kernel types
kernel_types = {'linear':'-t 0','polynomial':'-t 1','radial basis function':'-t 2'}

perprint("\tUsing hard-SVM (cost c= 10) to compare 3 kernel functions in default mode: \n")
perprint("\t0 -- linear: u'*v ")
perprint("\t1 -- polynomial: (gamma*u'*v + coef0)^degree ")
perprint("\t2 -- radial basis function: exp(-gamma*|u-v|^2) ")
perprint("\t(-d degree: default = 3), (-r coef0: default 0) (-g default = 1/num_features)\n")
accuracies = []
for k, param in kernel_types.items():
    # Set SVM parameters
    parameters = f'-s 0 {param} -c 10 -q' 
    """ -q for quite mode. 
        (-c value = large_value) to use hard margin
        (-d default = 3) in polynomial kernel
        (-r coef0: default 0) in polynomial kernel
        (-g default = 1/num_features) in rbf and polynomial kernels
        (-c default = 1)
    """
    
    # Train the SVM model: Y comes before X
    model = svm_train(Y_train, X_train, parameters) 
    p_label, p_acc, p_vals = svm_predict(Y_test, X_test, model, '-q')

    perprint(f"\t{k} kernel's accuracy: {p_acc[0]:.2f}% ")
    accuracies.append(p_acc[0])

""" Optimal kernel """
argmax_index = accuracies.index(max(accuracies))
k, param = list(kernel_types.items())[argmax_index]
perprint(f"\n\t{k} kernel is best under default conditions, \n\twith accuracy: {accuracies[argmax_index]:.2f}% ")



###
perprint("\n   Task 2: \n")

# Define Grid Search
def Grid_Search(costs, gammas, degrees, coefficients, kernel, param, X_train, Y_train):
    if kernel == 'linear': 
        accuracies = np.zeros( len(costs) )

        for i in range(len(costs)):
            parameters = f'-s 0 {param} -v 3 -c {costs[i]} -q'
            accuracies[i] = svm_train(Y_train, X_train, parameters)

        accuracy_optimal = np.max(accuracies)
        argmax_index = np.argmax(accuracies)
        cost_optimal = costs[argmax_index]
        k, param = list(kernel_types.items())[argmax_index]

        return cost_optimal, accuracy_optimal

    elif kernel == 'radial basis function':   
        accuracies = np.zeros(( len(costs), len(gammas) ))

        for i in range(len(costs)):
            for j in range(len(gammas)):
                parameters = f'-s 0 {param} -v 3 -c {costs[i]} -g {gammas[j]} -q'
                accuracies[i, j] = svm_train(Y_train, X_train, parameters)
        
        accuracy_optimal = np.max(accuracies)
        argmax_indices = np.unravel_index(np.argmax(accuracies), accuracies.shape)
        c_index, g_index = argmax_indices
        cost_optimal = costs[c_index]
        gamma_optimal = costs[g_index]

        return cost_optimal, gamma_optimal, accuracy_optimal
    
    elif kernel == 'polynomial':   
        accuracies = np.zeros(( len(costs), len(gammas), len(degrees), len(coefficients) ))

        for i in range(len(costs)):
            for j in range(len(gammas)):
                for k in range(len(degrees)):
                    for p in range(len(coefficients)):
                        parameters = f'-s 0 {param} -v 3 -c {costs[i]} -g {gammas[j]} -d {degrees[k]} -r {coefficients[p]} -q'
                        accuracies[i, j, k, p] = svm_train(Y_train, X_train, parameters)
        
        accuracy_optimal = np.max(accuracies)
        argmax_indices = np.unravel_index(np.argmax(accuracies), accuracies.shape)
        print(argmax_indices) ###
        c_index, g_index, d_index, r_index= argmax_indices
        cost_optimal = costs[c_index]
        gamma_optimal = costs[g_index]
        degree_optimal = degrees[d_index]
        coef_optimal = coefficients[r_index]

        return cost_optimal, gamma_optimal, degree_optimal, coef_optimal, accuracy_optimal
    
    else:
      perprint("\n\tYou didn't give a valid input. \n")


#kernel_types = {'linear':'-t 0','polynomial':'-t 1','radial basis function':'-t 2'}
kernel_types = {'linear':'-t 0', 'radial basis function':'-t 2', 'polynomial':'-t 1'}
costs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
degrees = [2, 3, 4]
coefficients = [0, 1, 2]
perprint("\tUsing Grid Search on soft-SVM to find best values of: \n\tcost 'c', gamma 'g', degree 'd' and coefficient 'r' \n")
perprint(f"\tcosts: {costs} \n\tgammas: {gammas} \n\tdegrees: {degrees} \n\tcoefficients: {coefficients} \n")


results = {
    'linear': {'accuracy': None, 'cost': None},
    'polynomial': {'accuracy': None, 'cost': None, 'gamma': None, 'degree': None, 'coefficient': None},
    'radial basis function': {'accuracy': None, 'cost': None, 'gamma': None}
}

for kernel, param in kernel_types.items():
    # Set SVM parameters
    parameters = f'-s 0 {param} -q' 
    """ -q for quite mode. 
        (-c value = large_value) to use hard margin
        (-d default = 3) in polynomial kernel
        (-r coef0: default 0) in polynomial kernel
        (-g default = 1/num_features) in rbf and polynomial kernels
        (-c default = 1)
    """
    
    if kernel == 'linear':
        cost_optimal, accuracy_optimal = Grid_Search(costs, gammas, degrees, coefficients, kernel, param, X_train, Y_train)
        perprint(f"\tFor {kernel} kernel, optimal accuracy = {accuracy_optimal:.2f} %, for: \n\t\toptimal cost = {cost_optimal} ")

        new_values = {'accuracy': accuracy_optimal, 'cost': cost_optimal}
        results[kernel].update(new_values)

    elif kernel == 'radial basis function':
        cost_optimal, gamma_optimal, accuracy_optimal = Grid_Search(costs, gammas, degrees, coefficients, kernel, param, X_train, Y_train)
        perprint(f"\tFor {kernel} kernel, optimal accuracy = {accuracy_optimal:.2f} %, for: \n\t\toptimal cost = {cost_optimal} ")
        perprint(f"\t\tand optimal gamma = {gamma_optimal} ")

        new_values = {'accuracy': accuracy_optimal, 'cost': cost_optimal, 'gamma': gamma_optimal}
        results[kernel].update(new_values)

    elif kernel == 'polynomial':
        cost_optimal, gamma_optimal, degree_optimal, coef_optimal, accuracy_optimal = Grid_Search(costs, gammas, degrees, coefficients, kernel, param, X_train, Y_train)
        perprint(f"\tFor {kernel} kernel, optimal accuracy = {accuracy_optimal:.2f} %, for: \n\t\toptimal cost = {cost_optimal} ")
        perprint(f"\t\toptimal gamma = {gamma_optimal} ")
        perprint(f"\t\toptimal degree = {degree_optimal} \n\t\tand optimal coefficient = {coef_optimal} ")

        new_values = {'accuracy': accuracy_optimal, 'cost': cost_optimal, 'gamma': gamma_optimal, 'degree': degree_optimal, 'coefficient': coef_optimal}
        results[kernel].update(new_values)


perprint("\n\tComputing accuracies on the testing data: \n")

#kernel_types = {'linear':'-t 0','polynomial':'-t 1','radial basis function':'-t 2'}
for k, param in kernel_types.items():
    perprint(k, "\n")
    # Set SVM parameters
    if k == 'linear':
        cost = results[k]['cost']
        parameters = f'-s 0 {param} -c {cost} -q' 
        model = svm_train(Y_train, X_train, parameters) 
        _, p_acc, __ = svm_predict(Y_test, X_test, model, '-q')
        results[k]['accuracy'] = p_acc[0]
        perprint(f"\tFor {k} kernel, optimal accuracy = {p_acc[0]:.2f} %, for: \n\t\toptimal cost = {cost} \n")

    elif k == 'polynomial':
        cost = results[k]['cost']
        gamma = results[k]['gamma']
        degree = results[k]['degree']
        coefficient = results[k]['coefficient']
        parameters = f'-s 0 {param} -c {cost} -g {gamma} -d {degree} -r {coefficient} -q' 
        model = svm_train(Y_train, X_train, parameters) 
        _, p_acc, __ = svm_predict(Y_test, X_test, model, '-q')
        results[k]['accuracy'] = p_acc[0]
        perprint(f"\tFor {k} kernel, optimal accuracy = {p_acc[0]:.2f} %, for: \n\t\toptimal cost = {cost} ")
        perprint(f"\t\toptimal gamma = {gamma} ")
        perprint(f"\t\toptimal degree = {degree} \n\t\tand optimal coefficient = {coefficient} \n")

    elif k == 'radial basis function':
        cost = results[k]['cost']
        gamma = results[k]['gamma']
        parameters = f'-s 0 {param} -c {cost} -g {gamma} -q' 
        model = svm_train(Y_train, X_train, parameters) 
        _, p_acc, __ = svm_predict(Y_test, X_test, model, '-q')
        results[k]['accuracy'] = p_acc[0]
        perprint(f"\tFor {k} kernel, optimal accuracy = {p_acc[0]:.2f} %, for: \n\t\toptimal cost = {cost} ")
        perprint(f"\t\toptimal gamma = {gamma} \n")


print("\tRanking all the kernels after grid search: ")
# Sort the kernels based on accuracy
sorted_kernels = sorted(results.keys(), key = lambda k: results[k]['accuracy'], reverse=True)

# Print the sorted kernels with ranking numbers and their accuracies
for rank, kernel in enumerate(sorted_kernels, start=1):
    print(f"\t\tRank {rank}: {kernel}: Accuracy - {results[kernel]['accuracy']:.2f}")


import pickle
# Pickle file path
pickle_path = "/content/drive/MyDrive/NYCU/00- Sem 01 - Fall 2023/Machine Learning/HMW_05/grid_search_results.pkl"
# Save the dictionary to a file
with open(pickle_path, 'wb') as file:
    pickle.dump(results, file)
# Load the dictionary from the file
with open(pickle_path, 'rb') as file:
    loaded_results = pickle.load(file)

# Numpy file path
numpy_path = "/content/drive/MyDrive/NYCU/00- Sem 01 - Fall 2023/Machine Learning/HMW_05/grid_search_results.npy"
# Save the dictionary to a file
np.save(numpy_path, results)
# Load the dictionary from the file
loaded_results = np.load(numpy_path, allow_pickle = True).item()




###
perprint("\n   Task 3: \n")


def linearKernel(X1, X2):
    kernel = X1 @ X2.T
    return kernel
    
def RBFKernel(X1, X2, gamma):
    dist = ssd.cdist(X1, X2, 'sqeuclidean') 
    kernel = np.exp(-1 * gamma * dist)
    return kernel

def precomputed_kernel(X1, X2, gamma):
    kernel_linear = linearKernel(X1, X2)
    kernel_RBF = RBFKernel(X1, X2, gamma)

    kernel = kernel_linear + kernel_RBF
    kernel = np.hstack(( np.arange(1, len(X1)+1).reshape(-1,1), kernel ))
    return kernel

perprint("\tUsing 'linear kernel + RBF kernel' to form a new kernel: \n")

#results['radial basis function']['gamma'] = 0.01
gamma = results['radial basis function']['gamma']

# Train SVM with new kernel
kernel_train = precomputed_kernel(X_train, X_train, gamma)
prob = svm_problem(Y_train, kernel_train, isKernel = True)

param = svm_parameter('-q -t 4') # 4 for precomputed kernels
model = svm_train(prob, param)

# Test SVM with new kernel
kernel_test = precomputed_kernel(X_test, X_train, gamma)
p_label, p_acc, p_vals = svm_predict(Y_test, kernel_test, model, '-q')

perprint(f"\t'linear + RBF' kernel's accuracy: {p_acc[0]:.2f}%  \n")


# Adding a new kernel in my dictionary
new_kernel = 'linear + RBF'
results[new_kernel] = {'accuracy': p_acc[0], 'gamma': gamma}

print("\tRanking all the kernels: ")
# Sort the kernels based on accuracy
sorted_kernels = sorted(results.keys(), key = lambda k: results[k]['accuracy'], reverse=True)

# Print the sorted kernels with ranking numbers and their accuracies
for rank, kernel in enumerate(sorted_kernels, start=1):
    print(f"\t\tRank {rank}: {kernel}: Accuracy - {results[kernel]['accuracy']:.2f}")



perprint("\n\nEnd")
# Close the RTF file at the end of your code
rtf_file.close()