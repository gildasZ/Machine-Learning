'''
HOMEWORK 03
'''

import numpy as np 
import matplotlib.pyplot as plt
import re

###
# Define the RTF file path
#rtf_file_path = "HMW_03_output.txt"
rtf_file_path = "HMW_03_output.rtf"

# Open the RTF file in write mode and clear its contents (if it exists)
rtf_file = open(rtf_file_path, 'w')
rtf_file.close()  # This will clear the contents if the file already exists

# Reopen the RTF file in append mode to keep it open
rtf_file = open(rtf_file_path, 'a')

# Define a custom "pseudo print" function
def pseudo_print(message, end = None, file = rtf_file):
    print(message, end = end)
    print(message, end = end, file = file)
###

pseudo_print("Starting...\n")

#
##
### 1) a. Univariate gaussian data generator
pseudo_print("1) a. Univariate gaussian data generator. \n")
##
def UnivariateGaussianDataGenerator(m, s):
    # Using the Box-Muller Method

    # Get U and V from the standard uniform distribution: U(0, 1) 
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    #print (U, "  ", V)

    # Sample Standard Normal distribution by Box-Muller Method
    Z = np.sqrt(- 2 * np.log(U)) * np.cos(2 * np.pi * V)

    # Sample the General Gaussian distribution
    X = m + np.sqrt(s) * Z

    return X

while True:
    #break ###
    user_input = input("Enter the mean and the variance separated by a space, \n(or type 'exit' to stop): ")

    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        #take_input = False
        break
    else:
        # Split the input string into two parts
        inputs = user_input.split()

        # Ensure there are exactly two inputs
        if len(inputs) != 2:
            pseudo_print("Invalid input.Please enter valid floating-point numbers.")
        else:
            try:
                # Convert the input parts to floats
                number1 = float(inputs[0])
                number2 = float(inputs[1])
                pseudo_print(f"You entered two floating-point numbers: {number1} and {number2}")
                sample = UnivariateGaussianDataGenerator(number1, number2)
                pseudo_print(f"Generated univariate Gaussian data point: {sample:.15} \n")
                break
            except ValueError:
                pseudo_print("Invalid input. Please enter valid floating-point numbers.")

#
##
### 1) b. Polynomial basis linear model data generator
pseudo_print("1) b. Polynomial basis linear model data generator. \n")
##
def Polynomial_basis_linear_model_data_generator(n, a, w):
    # x0 is uniformly distributed, x0 ~ -1.0 < x0 < 1.0
    lower_bound = - (1 - 1e-15)
    x0 = np.random.uniform(lower_bound, 1)
    
    # List comprehension to get a list of powers of x0
    x = [np.power(x0,i) for i in range(len(w))]

    # Get e with the univariate Gaussian data generator
    e = UnivariateGaussianDataGenerator(0, a)

    # Compute the value of y, w*x will perform element-wise multiplication
    y = np.sum(w*x) + e
    
    return x0, y

while True:
    #break ###

    n = 2  # Polysonial basis number
    a = 10 # Variance of distribution N(0, a)
    w = np.asarray([2,5]) # Not same as np.array([2,5])
    pseudo_print(f"Inputs: n(basis number) = {n}, a(variance) = {a}, and w = {w}")

    x0, y = Polynomial_basis_linear_model_data_generator(n, a, w)
    pseudo_print(f"Randomly selected x = {x0} \nOutput: y = {y:.15} \n")

    break

#
##
### 2) Sequential Estimator
pseudo_print("2) Sequential Estimator. \n")
##
m = 3.0
s = 5.0
pseudo_print(f"Data point source function: N({m}, {s}) \n")

eps = 1e-1
mean_curr = 0 # Current Mean
pop_var_curr = 0 # We use the population variance for calculations
n = 0 # Current total count

# Welford's online algorithm for updating variance
while (eps < abs(mean_curr-m) or eps < abs(pop_var_curr-s)) and n < 50000 :
    #break ###
    # Generate new data based on fixed mean (m) and fixed variance (s)
    new_point = UnivariateGaussianDataGenerator(m, s)
    pseudo_print(f"Add data point: {new_point:.16f}")
    # Update the total number of data points
    n += 1

    if n == 1:
        mean_curr = new_point
        pop_var_curr = 0
        pseudo_print(f"Mean = {mean_curr:20.16f}   Variance =   {pop_var_curr:.1f}")
    else:
        # Estimate the new mean
        mean_previous = mean_curr
        mean_curr = mean_curr + (new_point - mean_curr) / n

        # Estimate the new sample variance: After testing, the biased variance formula is 
        # what is used for sample input & output of question 2, so entire population variance

        #s_var_curr = s_var_curr + (new_point - mean_previous)**2 / n - s_var_curr / (n-1)
        #s_var_curr = s_var_curr * (n-2)/(n-1) + (new_point - mean_previous)**2 / n
        #pop_var_curr = pop_var_curr + (1/n) * ( (new_point - mean_previous)*(new_point - mean_curr) - pop_var_curr )
        pop_var_curr = (1/n) * ( (n-1) * pop_var_curr + (new_point - mean_previous)*(new_point - mean_curr) )
        pseudo_print(f"Mean = {mean_curr:20.16f}   Variance = {pop_var_curr:20.16f}")

pseudo_print(f"\nTotal sampling points: {n} points. \n")
    
#
##
### 3) Baysian Linear regression
pseudo_print("3) Baysian Linear regression. \n")
##
# Function for printing at each case of question 3
def plotting(subplot, num_points, n, x, weight, variance, a_y, point_x, point_y, title, ground_truth=False):
    #print(f"Checking this n value: {n}")###

    abs_errors = np.zeros(len(x)) # prediction of y
    y_values = np.zeros(len(x))
    for i in range(len(x)):
        X_hold = np.asarray( [np.power(x[i], k) for k in range(n)] ).reshape(1, -1) # 1 x n
        y_values[i] =  X_hold @ weight 
    #y_values = mean_predict # Same as mean_predict
    #function = np.poly1d(np.flip(weight))
    #y_values = function(x_values)

    if ground_truth == True:
        abs_errors = np.full(len(x), a) # error is the variance of the error in y formula
        
    else:
        for i in range(len(x)):
            X_hold = np.asarray( [np.power(x[i], k) for k in range(n)] ).reshape(1, -1) # 1 x n
            variance_predict= ((1/a_y) + X_hold @ variance @ X_hold.T).item()
            abs_errors[i] = variance_predict
        
        subplot.plot(point_x[:num_points], point_y[:num_points], 'bo')

    subplot.plot(x, y_values, 'k-')
    subplot.plot(x, y_values + abs_errors, 'r-')
    subplot.plot(x, y_values - abs_errors, 'r-')
    subplot.set_xlim(-2, 2) 
    subplot.set_ylim(-20, 25) 
    subplot.set_title(title) 

# Function for handling each case of question 3
def Compute_Print_Case(b, n, a, w):

    pseudo_print(f"b = {b}, n = {n}, a = {a}, w = [", end = '')
    for i in range(n-1):
        pseudo_print(f"{w[i]}, ", end = '')
    pseudo_print(f"{w[n-1]}] \n")

    # Sampling 1000 points
    NUM = 1000
    point_x=[]
    point_y=[]

    # [after 10 points, after 50 points, final result]
    mean_list = []
    variance_list = []

    mean = np.zeros((n, 1)) 
    variance = (1 / b) * np.identity(n)
    threshold = 1e-3
    iterations = 0
    design_mat = np.zeros((1, n))
    y_mat = np.zeros((1, 1))

    the_fig, my_axes = plt.subplots(nrows=2, ncols=2)


    while iterations <= NUM: #eps < np.linalg.norm(w-mean)
        iterations += 1
        #add point
        point = Polynomial_basis_linear_model_data_generator(n, a, w)
        #point = new_point
        pseudo_print(f"Add data point ({point[0]:.5f}, {point[1]:.5f}):")
        
        #update mean & variance
        X = np.asarray([np.power(point[0], i) for i in range(n)]).reshape(1, -1) # 1 x n
        y = point[1]
        
        if iterations == 1:
            design_mat = X
            y_mat[0] = y
            #print(y_mat)
        else:
            design_mat = np.vstack((design_mat, X))
            y_mat = np.vstack((y_mat, np.array(y)))
            #print(y_mat)

        a_y = 1/a # Precision of y distribution, constant
        S = np.linalg.pinv(variance)
        #variance_new = np.linalg.pinv(a_y * X.T @ X + S)  # Should I not use design matrix? How about a? a^-1 instead?
        variance_new = np.linalg.pinv(a_y * design_mat.T @ design_mat + S)  # Should I not use design matrix? How about a? a^-1 instead?
        #print(variance_new)

        #mean_new = variance_new @ (a_y * X.T * y + S @ mean) # Where did this formula come from?
        #mean_new = a_y * variance_new @ X.T * y
        mean_new = variance_new @ (a_y * (design_mat.T) @ y_mat + S @ mean)
        #print(mean_new)

        pseudo_print("\nPosterior mean: ")
        for i in range(n):
            pseudo_print(f"{mean_new[i].item():15.10f}")

        pseudo_print("\nPosterior variance: ")
        for i in range(n):
            for j in range(n-1):
                pseudo_print(f"{variance_new[i][j].item():15.10f},", end = '')
            pseudo_print(f"{variance_new[i][n-1].item():15.10f}")
        
        # Predictive distribution: Will use the mean and variance before the new point
        predictive_mean = (X @ mean).item() # .item() will return the single element as a Python scalar
        ##       Check above    ####

        predictive_variance = ((1/a_y) + X @ variance @ X.T).item()
        ##       Check above    ####

        pseudo_print(f"\nPredictive distribution ~ N({predictive_mean:.5f}, {predictive_variance:.5f}) \n")
        pseudo_print('--------------------------------------------------')
        
        #save record
        point_x.append(point[0])
        point_y.append(point[1])

        if iterations == 1:
            x_values = np.linspace(-2, 2, 500)
            title = "Ground Truth" # Set ground_truth=True
            plotting(my_axes[0][0], iterations, n, x_values, w, variance, a_y, point_x, point_y, title, ground_truth=True)
            #break
        elif iterations == 10:
            mean_list.append(mean_new)
            variance_list.append(variance_new)

            x_values = np.linspace(-2, 2, 500)
            title = f"After {iterations} incomes"
            # Will plot two times
            plotting(my_axes[1][0], iterations, n, x_values, mean_list[-1], variance_list[-1], a_y, point_x, point_y, title, ground_truth=False)
            
        elif iterations == 50:
            mean_list.append(mean_new)
            variance_list.append(variance_new)

            x_values = np.linspace(-2, 2, 500)
            title = f"After {iterations} incomes"
            # Will plot two times
            plotting(my_axes[1][1], iterations, n, x_values, mean_list[-1], variance_list[-1], a_y, point_x, point_y, title, ground_truth=False)
             
        elif iterations == NUM:
            mean_list.append(mean_new)
            variance_list.append(variance_new)

            x_values = np.linspace(-2, 2, 500)
            title = f"Predict result ({iterations} incomes)"
            # Maybe will plot
            plotting(my_axes[0][1], iterations, n, x_values, mean_list[-1], variance_list[-1], a_y, point_x, point_y, title, ground_truth=False)
            plt.show()
            #plt.savefig('combined_figure.png')
            break

        mean = mean_new
        variance = variance_new

while True:
    #break
    # Case 1    
    b = 1 # Initial prior gaussian distribution's variance
    n = 4 # Polynomial basis, also dimension of d of design matrix
    a = 1 # variance of N(0, a)
    w = np.asarray([1, 2, 3, 4]).reshape(-1, 1) # line parameters

    pseudo_print("Case 1. ", end = '')
    Compute_Print_Case(b, n, a, w)

    # Case 2    
    b = 100 # Initial prior gaussian distribution's variance
    n = 4 # Polynomial basis, also dimension of d of design matrix
    a = 1 # variance of N(0, a)
    w = np.asarray([1, 2, 3, 4]) # line parameters

    pseudo_print("Case 2. ", end = '')
    Compute_Print_Case(b, n, a, w)

    # Case 3    
    b = 1 # Initial prior gaussian distribution's variance
    n = 3 # Polynomial basis, also dimension of d of design matrix
    a = 3 # variance of N(0, a)
    w = np.asarray([1, 2, 3]) # line parameters
    
    pseudo_print("Case 3. ", end = '')
    Compute_Print_Case(b, n, a, w)

    break

pseudo_print("\nEnd")

# Close the RTF file at the end of your code
rtf_file.close()