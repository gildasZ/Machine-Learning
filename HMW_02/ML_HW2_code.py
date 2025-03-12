'''
HOMEWORK 02
'''
#
#
# To work in discrete mode enter 0, 
# To work in continuous mode enter 1, 
discrete_mode = 1
#

import sys, math, binascii
import numpy as np
from pathlib import Path 

# Uploading txt file for question 1
path_train_images = Path(r"HMW_02\Downloaded_Files_Unzipped\train-images.idx3-ubyte")
path_train_labels = Path(r"C:HMW_02\Downloaded_Files_Unzipped\train-labels.idx1-ubyte")
path_test_images = Path(r"C:HMW_02\Downloaded_Files_Unzipped\t10k-images.idx3-ubyte")
path_test_labels = Path(r"C:HMW_02\Downloaded_Files_Unzipped\t10k-labels.idx1-ubyte")

# Uploading txt file for question 2
data_file_q2_path = Path(r'HMW_02\Cat_Testfile_hmw02.txt')

def LoadImage(file_path):
        image_file = open(file_path, 'rb')
        image_file.read(4) # magic number
        return image_file

def LoadLabel(file_path):
    label_file = open(file_path, 'rb')
    label_file.read(4) # magic number
    return label_file

def PrintResult(prob, answer):
    print('Posterior (in log scale):')
    for value in range(prob.shape[0]):
        print(f'{value}: {prob[value]}')
    pred = np.argmin(prob)
    print(f'Prediction: {pred}, Ans: {answer}\n')
    return 0 if answer == pred else 1

def DrawImagination(image, image_row, image_col, mode):
    print('Imagination of numbers in Baysian classifier\n')
    if mode == 0: # discrete
        for digit in range(10):
            print(f'{digit}:')
            for j in range(image_row):
                for k in range(image_col):
                    white = sum(image[digit][j * image_row + k][:17])
                    black = sum(image[digit][j * image_row + k][17:])
                    print(f'{1 if black > white else 0} ', end='')
                print()
            print()
    elif mode == 1: # continuous
        for digit in range(10):
            print(f'{digit}:')
            for j in range(image_row):
                for k in range(image_col):
                    print(f'{1 if image[digit][j * image_row + k] > 128 else 0} ', end='')
                print()
            print()

def LoadTrainingData():
    train_image_file = LoadImage(path_train_images)
    train_size = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_row = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_col = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    train_label_file = LoadLabel(path_train_labels)
    train_label_file.read(4)
    return train_image_file, train_label_file, train_size, image_row, image_col

def LoadTestingData():
    test_image_file = LoadImage(path_test_images)
    test_size = int(binascii.b2a_hex(test_image_file.read(4)))
    test_image_file.read(4)
    test_image_file.read(4)
    test_label_file = LoadLabel(path_test_labels)
    test_label_file.read(4)
    return test_image_file, test_label_file, test_size

def DiscreteMode():
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    image_size = image_row * image_col
    image = np.zeros((10, image_size, 32), dtype=np.int32)
    image_sum = np.zeros((10, image_size), dtype=np.int32)
    prior = np.zeros((10), dtype=np.int32)

    for i in range(train_size):
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        prior[label] += 1
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            image[label][j][grayscale // 8] += 1 # Bin tag
            image_sum[label][j] += 1

    # testing
    error = 0
    for i in range(test_size):
        # print(i, error)
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=np.float)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16) # Load pixel values
        for j in range(10):
            # consider prior
            prob[j] += np.log(prior[j] / train_size) # log of class probability
            for k in range(image_size):
                # consider likelihood
                likelihood = image[j][k][test_image[k] // 8] # From train data, Given class j, given pixel position k, and its bin count
                if likelihood == 0:
                    likelihood = np.min(image[j][k][np.nonzero(image[j][k])]) # if 0, take min non zero bin count from train data
                # likelihood = 0.000001 if likelihood == 0 else likelihood
                prob[j] += np.log(likelihood / image_sum[j][k])  # Log of Bin count for pixel / sum all bin count for pixel
        # normalize
        summation = sum(prob)
        prob /= summation
        error += PrintResult(prob, answer)

    DrawImagination(image, image_row, image_col, 0)
    print(f'Error rate: {error / test_size}')

def ContinuousMode():
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    image_size = image_row * image_col    
    prior = np.zeros((10), dtype=np.int32)
    var = np.zeros((10, image_size), dtype=np.float)
    mean = np.zeros((10, image_size), dtype=np.float)
    mean_square = np.zeros((10, image_size), dtype=np.float)

    for i in range(train_size):
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        prior[label] += 1 # For now, gives count in the class
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            mean[label][j] += grayscale # Later will divide by count in the class
            mean_square[label][j] += (grayscale ** 2)
    
    for i in range(10):
        for j in range(image_size):
            mean[i][j] /= prior[i]
            mean_square[i][j] /= prior[i]
            var[i][j] = mean_square[i][j] - (mean[i][j] ** 2) # (x**2 - mean**2)/ count in class, not (x-mean)**2/count in class
            var[i][j] = 1000 if var[i][j] == 0 else var[i][j] # will later divide by the count in class

    # testing
    error = 0
    for i in range(test_size):
        # print(i, error)
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=np.float)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)
        for j in range(10):
            # consider prior
            prob[j] += np.log(prior[j] / train_size)
            for k in range(image_size):
                # consider likelihood
                likelihood = -0.5 * (np.log(2 * math.pi * var[j][k]) + ((test_image[k] - mean[j][k]) ** 2) / var[j][k])
                prob[j] += likelihood
        # normalize
        summation = sum(prob)
        prob /= summation
        error += PrintResult(prob, answer)

    DrawImagination(mean, image_row, image_col, 1)
    print(f'Error rate: {error / test_size}')

if __name__ == '__main__':
    mode = input('Discrete(0) or continuous(1): ')
    print(mode)
    if mode == '0':
        try:
            with open('Discrete', 'r', encoding='utf-8') as file:
                for line in file:
                    print(line, end='')
        except:
            DiscreteMode()
    elif mode == '1':
        try:
            with open('Continuous', 'r', encoding='utf-8') as file:
                for line in file:
                    print(line, end='')
        except:
            ContinuousMode()

#
#
#
#### Question 2: Online Learning
####
#
# Read the file line by line, parsing the data and storing in a list of NumPy arrays
data_list = []

with open(data_file_q2_path, 'r') as file:
    for line in file:
        parts = line.strip().split()  # Split the line into parts (space-separated)

        if len(parts) >= 2:
            # Extract the line number (index 0) and binary outcomes (from index 1 onwards)
            line_num = int(parts[0])
            binary_outcomes = tuple(list(map(int, ''.join(parts[1:]))))  # Convert binary string to NumPy array

            # Append the line data as a NumPy array to the list
            data_list.append((line_num, binary_outcomes))

# Now, data_list contains each line as a NumPy array of binary outcomes, along with its line number
# You can access individual lines as data_list[i] where i is the line number - 1 (zero-based index)

def binomial_likelihood_beta_prior_beta_posterior(data_list, a, b):
    """
    Calculate the Binomial likelihood of observing a given dataset...

    Parameters:
    - data: The binary outcomes dataset (e.g., 0s and 1s).
    """
    for data in data_list:

        print(f"\tCase {data[0]}: {''.join(map(str, data[1]))}")
        n = len(data[1])
        x = sum(data[1])
        binomial_coef = np.math.factorial(n)/(np.math.factorial(x)*np.math.factorial(n-x))
        p = x / n
        likelihood = binomial_coef*(p**x)*(1-p)**(n-x)

        #print(f"n: {n}, x: {x}, \nbinomial_coef: {binomial_coef}, p: {p}") ###
        print(f"\tLikelihood: {likelihood:.17}")
        print(f"\tBeta prior:     a ={a:3}  b ={b:3}")
        a += x
        b += n - x
        print(f"\tBeta posterior: a ={a:3}  b ={b:3}")
        print()

# Initializing the parameters for the beta prior 
a, b = 0, 0
print("\n\nCase 1: a = 0, b = 0 \n")
binomial_likelihood_beta_prior_beta_posterior(data_list, a, b)

a, b = 10, 1
print("Case 2: a = 10, b = 1 \n")
binomial_likelihood_beta_prior_beta_posterior(data_list, a, b)


print("End")