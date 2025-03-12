'''
HOMEWORK 04
'''

import numpy as np 
import matplotlib.pyplot as plt
import sys, math, binascii
from pathlib import Path 

###
# Define the RTF file path
#rtf_file_path = "HMW_04_output.txt"
rtf_file_path = "HMW_04_output.txt"

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
### 1) Logistic regression.
pseudo_print("1) Logistic regression. \n")
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

def generate_data_points(number, mean_x, mean_y, variance_x, variance_y):
    data = np.empty((number, 2))
    for i in range(number):
        data[i, 0] = UnivariateGaussianDataGenerator(mean_x, variance_x)
        data[i, 1] = UnivariateGaussianDataGenerator(mean_y, variance_y)
    return data

def gradient_descent(design_matrix, True_labels, alpha):
    #weight = np.random.rand(3, 1) # Initial weight
    weight = np.zeros((3, 1)) # Initial weight
    length = len(True_labels)
    condition = 100
    limit = 100
    convergence = 1e-1
    runs = 0
    while np.sqrt(condition) > convergence:
        #break
        check = np.dot(design_matrix, weight)
        check[check < -limit] = -limit
        sigmoid_values = 1 / (1 + np.exp(-check)) 
        '''
        sigmoid_values = np.empty((length, 1))
        for i in range(length):
            check = np.dot(design_matrix[i], weight)
            if check < - limit:
                check = -limit
            tempo = 1/ ( 1 + np.exp( - check) )
            sigmoid_values[i, 0] = tempo
        '''
        gradient = np.dot( design_matrix.T, ( sigmoid_values  - True_labels ) )  # Gradient matrix

        weight = weight - alpha * gradient
        condition = np.sum(gradient**2)
        runs += 1
    pseudo_print(f"Gradient descent converges after {int(runs)} runs.")
    return weight

def newton_method(design_matrix, True_labels, alpha):
    #weight = np.random.rand(3, 1) # Initial weight
    weight = np.zeros((3, 1)) # Initial weight
    eps = 1e-2
    N = len(design_matrix)
    D = np.zeros((N, N))
    condition = 100
    limit = 100
    runs = 0

    while np.sqrt(condition) > eps:
        #break
        #sigmoid_values = np.empty((N, 1))
        check = np.dot(design_matrix, weight)
        check[check < -limit] = -limit
        sigmoid_values = 1 / (1 + np.exp(-check)) # sigmoid_values

        gradient = np.dot( design_matrix.T, ( sigmoid_values - True_labels ) ) # Gradient matrix
        D = np.diagflat(sigmoid_values * (1 - sigmoid_values)) # Diagonal matrix
        H = np.dot( np.dot(design_matrix.T, D) , design_matrix ) # Hessian matrix

        try:
            H_inv = np.linalg.inv(H)
            weight = weight - np.dot ( H_inv, gradient )
        except np.linalg.LinAlgError as error:
            print(str(error))
            print(runs+1, '\t', 'Hessian matrix non invertible, switch to Gradient descent')
            weight = weight - alpha * gradient

        condition = np.sum(gradient**2)
        runs += 1
    pseudo_print(f"Newton's method converges after {int(runs)} runs.")

    return weight

def predict(design_matrix, weight):
    length = len(design_matrix)
    limit = 100
    #predictions = np.empty((length, 1))
    check = np.dot(design_matrix, weight)
    predictions = np.where(check < 0, 0, 1) # predictions[i] = 0 if check[i] < 0 else 1
    '''
    for i in range(length):
        hold = np.dot(design_matrix[i], weight)
        sigmoid = 1/(1 + np.exp(-hold))
        #predictions[i] = 0 if sigmoid < 0.5 else 1
        predictions[i] = 0 if hold < 0 else 1
    '''
    #pseudo_print(predictions.shape)
    return predictions

def ploting(subplot, x_values, y_values, color_type, title):
    subplot.plot(x_values, y_values, color_type) # "k." for black point, 'b.' 'r.' for blue and red ones
    subplot.set_xlim(-5, 15) 
    subplot.set_ylim(-2.5, 14.5) 
    subplot.set_title(title) 

def confusion_matrix(design_matrix, True_labels, predictions):
    '''
    let class0 be positive, class1 be negative
    ----------
    | TN  FP |  <= confusion matrix (Look at wiki)
    | FN  TP |
    ----------
    :design_matrix: (2N, 3) shaped matrix
    :True_labels: (2N, 1) shaped matrix
    :predictions: (2N,1) shaped matrix
    :return: (confusion_matix, points to be class0, points to be class1)
    '''
    length = len(design_matrix)
    TP = FP = FN = TN = 0

    acc = 0
    for i in range(length):
        if True_labels[i] == predictions[i] == 1:
            TP += 1
            acc += 1
        elif True_labels[i] == predictions[i] == 0:
            TN += 1
            acc += 1
        elif True_labels[i] == 0 and predictions[i] == 1:
            FP += 1
        else:
            FN += 1

    #pseudo_print(f"Number accurate = {acc} \n")

    matrix = np.empty((2,2))
    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = TN, FP, FN, TP

    D1_predict = []
    D2_predict = []
    for i in range(length):
        if predictions[i] == 0:
            D1_predict.append(design_matrix[i, 1:])
        else:
            D2_predict.append(design_matrix[i, 1:])

    return matrix, np.array(D1_predict).reshape(-1, 2), np.array(D2_predict).reshape(-1, 2)

def print_elements(subplot, title, weight, design_matrix, True_labels):

    for i in range(len(weight)):
        pseudo_print(f"{weight[i].item():15.10f}")
    pseudo_print("")
    
    pseudo_print("Confusion Matrix:  ")
    predictions = predict(design_matrix, weight)
    conf_matrix, D1_pred, D2_pred = confusion_matrix(design_matrix, True_labels, predictions)

    pseudo_print("\t     Predict cluster 1 Predict cluster 2")   
    pseudo_print(f"Is cluster 1        {int(conf_matrix[0, 0])} \t\t     {int(conf_matrix[0, 1])}")
    pseudo_print(f"Is cluster 2        {int(conf_matrix[1, 0])} \t\t     {int(conf_matrix[1, 1])}")

    sensitivity = conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[1, 0]) # TRUE POSITIVE RATE
    specificity = conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[0, 1]) # TRUE NEGATIVE RATE
    pseudo_print(f"\nSensitivity (Successfully predict cluster 1): {specificity}") # Correct as per assignment
    pseudo_print(f"Specificity (Successfully predict cluster 2): {sensitivity}") # Correct as per assignment
    pseudo_print("\n_____________________________________________")

    color_type = 'r.'
    ploting(subplot, D1_pred[:, 0], D1_pred[:, 1], color_type, title)
    color_type = 'b.'
    ploting(subplot, D2_pred[:, 0], D2_pred[:, 1], color_type, title)

def handle_given_case(N, mx1, my1, vx1, vy1, mx2, my2, vx2, vy2, alpha = 0.001):
    
    fig_results, axes = plt.subplots(nrows = 1, ncols = 3)
    D1 = generate_data_points(N, mx1, my1, vx1, vy1)
    D2 = generate_data_points(N, mx2, my2, vx2, vy2)

    label_D1 = np.zeros((N, 1))
    label_D2 = np.ones((N, 1))
    Col_ones = np.ones((N, 1))

    Training_X_and_Y = np.vstack(( np.hstack((Col_ones, D1, label_D1)), np.hstack((Col_ones, D2, label_D2)) ))
    Training_X_and_Y = Training_X_and_Y[np.random.permutation(2*N)]

    design_matrix = Training_X_and_Y[:, :3]
    True_labels = Training_X_and_Y[:, 3].reshape(-1, 1)
    
    ############################
    title = "Ground truth"
    color_type = 'r.'
    ploting(axes[0], D1[:, 0], D1[:, 1], color_type, title)
    color_type = 'b.'
    ploting(axes[0], D2[:, 0], D2[:, 1], color_type, title)

    ############################
    # Logistic regression with gradient descent to separate D1 and D2
    pseudo_print("Gradient descent: \n")
    weight = gradient_descent(design_matrix, True_labels, alpha)
    pseudo_print("\nw: ")
    title = "Gradient descent"
    print_elements(axes[1], title, weight, design_matrix, True_labels)

    ############################
    # Logistic regression with gradient descent to separate D1 and D2
    pseudo_print("Newton's method: \n")
    weight = newton_method(design_matrix, True_labels, alpha)
    pseudo_print("\nw: ")
    title = "Newton's method"
    print_elements(axes[2], title, weight, design_matrix, True_labels)

    ############################
    plt.show()

while True:
    #break
    # Case 1
    N = 50
    mx1 = 1
    my1 = 1
    vx1 = 2
    vy1 = 2
    mx2 = 10
    my2 = 10
    vx2 = 2
    vy2 = 2
    alpha = 0.001
    pseudo_print(f"Case 1: N = {N}, mx1 = my1 = {mx1}, mx2 = my2 = {mx2}, vx1 = vy1 = vx2 = vy2 = {vx1} \n")
    handle_given_case(N, mx1, my1, vx1, vy1, mx2, my2, vx2, vy2, alpha)
    
    # Case 2
    N = 50
    mx1 = 1
    my1 = 1
    vx1 = 2
    vy1 = 2
    mx2 = 3
    my2 = 3
    vx2 = 4
    vy2 = 4
    alpha = 0.001
    pseudo_print(f"Case 2: N = {N}, mx1 = my1 = {mx1}, mx2 = my2 = {mx2}, vx1 = vy1 = vx2 = vy2 = {vx1} \n")
    handle_given_case(N, mx1, my1, vx1, vy1, mx2, my2, vx2, vy2, alpha)

    break

#
##
### 2) EM algorithm.
pseudo_print("\n2) EM algorithm. \n")
##

# Uploading paths for question 2
path_train_images = Path(r"HMW_04\Downloaded_Files_Unzipped\train-images.idx3-ubyte")
path_train_labels = Path(r"HMW_04\Downloaded_Files_Unzipped\train-labels.idx1-ubyte")

def LoadImage(file_path):
        image_file = open(file_path, 'rb')
        image_file.read(4) # magic number
        return image_file

def LoadLabel(file_path):
    label_file = open(file_path, 'rb')
    label_file.read(4) # magic number
    return label_file

def Load_Data():
    pseudo_print("Loading data...")
    train_image_file = LoadImage(path_train_images)
    train_size = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    #train_size = 10000
    image_row = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_col = int(binascii.b2a_hex(train_image_file.read(4)), 16)

    train_label_file = LoadLabel(path_train_labels)
    check = int(binascii.b2a_hex(train_label_file.read(4)), 16) # will read the number of items, same as train size
    if check != train_size:
        pseudo_print("Number train_images different from number train_labels")

    image_size = image_row * image_col
    train_images = np.empty((train_size, image_size))
    train_labels = np.empty((train_size, 1))

    for i in range(train_size):
        train_labels[i, 0] = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            train_images[i, j] = grayscale // 128  # Pixel value = 0 (0~127) or 1 (128~255)
    train_labels = train_labels.astype(int)
    train_images = train_images.astype(int)
    unique_elements = np.unique(train_labels)
    unique_classes = len(unique_elements)
    pseudo_print("Data loaded. \n")
    return train_images, train_labels, train_size, image_row, image_col, unique_classes

def plot_discrete(distribution, unique_classes, image_row, image_col):
    for c in range(unique_classes):
        pseudo_print(f"class: {c}")
        for i in range(image_row):
            for j in range(image_col):
                pseudo_print(1 if distribution[c, i * image_row + j] > 0.5 else 0, end =' ')
            pseudo_print("")
        pseudo_print("")

def plot_labeled_discrete(distribution, unique_classes, image_row, image_col):
    for c in range(unique_classes):
        pseudo_print(f"labeled class: {c}")
        for i in range(image_row):
            for j in range(image_col):
                pseudo_print(1 if distribution[c, i * image_row + j] > 0.5 else 0, end =' ')
            pseudo_print("")
        pseudo_print("")

def initialize_parameters(num_classes, num_features):
    # 1. Initialize Parameters:
    # means[j, k] represents the probability of pixel j being '1' in class k, Bernoulli parameter
    # weights is the probability of each class

    pseudo_print("Starting initialization...")
    # Initialize weights randomly
    low = 0.1
    high = 0.9
    # Initialize weights, a 1D array with random values in the specified range
    weights = np.random.uniform(low, high, num_classes)
    weights /= np.sum(weights)
    for i in range(num_classes):
        if weights[i] == 0:
            pseudo_print(f"weights[{i}] == 0")

    #common_value = 0.1
    # Initialize a 1D array with constant values
    #weights = np.full(num_classes, common_value)
    #weights /= np.sum(weights)
    pseudo_print("weights initialized.")

    # Initialize means, a 2D array with random values in the specified range
    low = 0.1
    high = 0.9
    means = np.random.uniform(low, high, (num_classes, num_features))
    for i in range(num_classes):
        for j in range(num_features):
            if means[i, j] == 0:
                pseudo_print(f"means[{i}, {j}] == 0")
    
    # Initialize means, a 2D array with constant values
    #common_value = 0.5
    #means = np.full((num_classes, num_features), common_value) # Also called means by others
    pseudo_print("means initialized. \n")
    return weights, means

def expectation_step(data, weights, means, num_classes):
    # 2. E-step (Expectation Step):
    # Assuming 'data' contains the binary images (0s and 1s) of shape (num_images, num_pixels)
    # Calculate the probability of each image belonging to each class given its pixel values using Bayes' rule
    # 'responsabilities' will store the probability of each image belonging to each class
    # Use it to assign each image to the most likely class based on the current parameters.

    # Calculate responsibilities
    length = data.shape[0]
    responsibilities = np.empty((length, num_classes))
    zero_s = 0
    not_zero_s = 0

    for cluster in range(num_classes):
        responsibilities[:, cluster] = weights[cluster] * np.prod(means[cluster]**data * (1 - means[cluster])**(1 - data), axis=1)

    # Normalize responsibilities
    #responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    
    for i in range(length):
        for k in range(num_classes):
            # Calculate the likelihood: product of probabilities of pixels being '1' or '0' in the class

            # P(Class k | Image i) = P(Image i | Class k) * P(Class k)
            responsibilities[i, k] = weights[k] * np.prod(means[k]**data[i] * (1 - means[k])**(1 - data[i]))
        if np.sum(responsibilities[i]) != 0:
            responsibilities[i] /= np.sum(responsibilities[i])
            not_zero_s += 1
        elif np.sum(responsibilities[i]) == 0: 
            #pseudo_print(f"expectation_step: np.sum(responsibilities[{i}] == 0")
            zero_s += 1
    #pseudo_print(f"expectation_step: not_zero_s = {not_zero_s}, zero_s = {zero_s} \n")
    
    return responsibilities

def maximization_step(data, responsibilities, num_classes):
    # 3. M-step (Maximization Step):
    # Update the parameters of the Bernoulli distributions (means) based on the assigned images.
    # Re-estimate the probabilities of each pixel being '1' or '0' for each class using the images assigned to each class.

    # Update weights
    #weights = np.sum(responsibilities, axis=0) / np.sum(responsibilities)
    weights = np.zeros(num_classes)
    Total = data.shape[0] # = np.sum(responsibilities) because np.sum(responsibilities[i]) = 1
    for cluster in range(num_classes):
        weights[cluster] = np.sum(responsibilities[:, cluster]) / Total
        if weights[cluster] == 0:
            pseudo_print(f"maximization_step: weights[{cluster}] == 0")
    
    # Update means
    epsilon = 0 # Add a pseudo count to the means to avoid issues 
    # with calculating the responsibilities later
    means = np.zeros((num_classes, data.shape[1]))
    for cluster in range(num_classes):
        means[cluster] = np.sum(responsibilities[:, cluster, np.newaxis] * data, axis=0) / np.sum(responsibilities[:, cluster])
        
        """
        zero_s = 0
        not_zero_s = 0
        denominator = np.sum(responsibilities[:, cluster]) + epsilon * data.shape[0]
        if denominator == 0: denominator = 1
        for j in range(data.shape[1]):
            numerator = epsilon + np.sum(responsibilities[:, cluster] * data[:, j])  ## Issue is here?
            means[cluster, j] = numerator / denominator

            if means[cluster, j] == 0:
                zero_s += 1
            else:
                not_zero_s += 1
        #pseudo_print(f"maximization_step: cluster[{cluster:2}] with {zero_s:3} zero_s means and {not_zero_s:3} not_zero_s means")
    #pseudo_print("")
        """
    return weights, means

def predict_clusters(responsibilities, unique_classes):
    # Assign data points to the most probable cluster
    predicted_clusters = np.argmax(responsibilities, axis=1)
    return predicted_clusters

def evaluate(true_labels, predicted_labels, class_value):
    '''
    Let Positive be ""Is class", and Negative be "Is not class"
    ----------
    | TP  FN |  <= confusion matrix (Look at wiki)
    | FP  TN |
    ----------
    :return: (confusion_matix, points to be class0, points to be class1)
    '''
    length = len(true_labels)
    matrix = np.zeros((2, 2))
    TP = FP = FN = TN = 0

    for i in range(length):
        # Predict P
        if predicted_labels[i] == class_value: 
            if true_labels[i] == class_value:
                TP += 1
            else:
                FP += 1
        # Predict not P -> N
        else:
            if true_labels[i] == class_value:
                FN += 1
            else:
                TN += 1
    try:
        sensitivity = TP / (TP + FN) # TRUE POSITIVE RATE
    except ZeroDivisionError:
        sensitivity = TP / 1# positive # (TP + FN) # TRUE POSITIVE RATE
    try:
        specificity = TN / (TN + FP) # TRUE NEGATIVE RATE
    except ZeroDivisionError:
        specificity = TN / 1# negative #(TN + FP) # TRUE NEGATIVE RATE

    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = TP, FN, FP, TN
    return matrix, sensitivity, specificity

def print_EM(confusion_matrix, sensitivity, specificity, class_value):
    matrix = confusion_matrix
    pseudo_print(f"Confusion Matrix {int(class_value)}")
    pseudo_print(f"\t        Predict number {int(class_value)}    Predict not number {int(class_value)}")   
    pseudo_print(f"Is number {int(class_value)}       {int(matrix[0, 0]):6} \t\t   {int(matrix[0, 1]):6}")
    pseudo_print(f"Is not number {int(class_value)}   {int(matrix[1, 0]):6} \t\t   {int(matrix[1, 1]):6}")
    pseudo_print(f"\nSensitivity (Successfully predict number {int(class_value)})    : {sensitivity:.5}") 
    pseudo_print(f"Specificity (Successfully predict not number {int(class_value)}): {specificity:.5}") 
    pseudo_print("\n_____________________________________________\n")

## Applying EM algo
def expectation_maximization():
    train_images, train_labels, train_size, image_row, image_col, unique_classes = Load_Data()
    image_size = image_row * image_col
    run_number = 0
    difference = 1000

    # 1. Initialize Parameters:
    weights, p = initialize_parameters(unique_classes, image_size)

    while run_number < 100 and difference > 20 :
        run_number += 1

        # 2. E-step (Expectation Step):
        # Assign each image to the most likely class based on the current parameters.
        # 'responsabilities' will store the probability of each image belonging to each class
        responsibilities = expectation_step(train_images, weights, p, unique_classes)
        
        # Make predictions based on the responsibilities (shape: train_size x unique_classes)
        predictions = predict_clusters(responsibilities, unique_classes)

        # 3. M-step (Maximization Step):
        # Update the parameters of the Bernoulli distributions based on the assigned images.
        # Re-estimate the probabilities of each pixel being '1' or '0' for each class using the images assigned to each class.

        # Update weights # They must sum to 1
        weights_new, p_new = maximization_step(train_images, responsibilities, unique_classes)
        
        # Plot imagination of numbers
        plot_discrete(p_new, unique_classes, image_row, image_col)
        difference = np.sum(np.sum(np.abs(p_new - p)) + np.sum(np.abs(weights_new - weights)))
        pseudo_print(f"No. of Iteration: {int(run_number)}, Difference: {difference:.15}\n")
        pseudo_print("_____________________________________________")
        pseudo_print("_____________________________________________\n")
        weights, p = weights_new, p_new

    labeled_means = np.zeros((unique_classes, image_size))
    for cluster in range(unique_classes):
        denom = 0
        for i in range(train_size):
            if train_labels[i] == cluster:
                denom += 1
                labeled_means[cluster] = labeled_means[cluster] + train_images[i]
        labeled_means[cluster] /= denom
    plot_labeled_discrete(labeled_means, unique_classes, image_row, image_col)
    pseudo_print("_____________________________________________")
    pseudo_print("_____________________________________________\n")

    for i in range(unique_classes):
        confusion_matrix, sensitivity, specificity = evaluate(train_labels, predictions, i)
        print_EM(confusion_matrix, sensitivity, specificity, i)

    pseudo_print(f"Total iteration to converge: {int(run_number)}")
    count_ = 0
    for i in range(train_size):
        if predictions[i] != train_labels[i]:
            count_ += 1
    errors = count_/train_size
    pseudo_print(f"Total error rate: {errors:.15}")


expectation_maximization()
pseudo_print("\nEnd")
# Close the RTF file at the end of your code
rtf_file.close()
