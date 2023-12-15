import numpy as np
from keras.datasets import cifar10, cifar100

# Load CIFAR10 and CIFAR100 data
(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()

# Select required data from CIFAR10 and CIFAR100
cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
cifar100_classes = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]

#Function for filtering CIFAR10 and CIFAR100 data
def filter_cifar10(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test):
    cifar10_x_train_filtered = cifar10_x_train[np.isin(cifar10_y_train, cifar10_classes).flatten()]
    cifar10_y_train_filtered = cifar10_y_train[np.isin(cifar10_y_train, cifar10_classes).flatten()]
    cifar10_x_test_filtered = cifar10_x_test[np.isin(cifar10_y_test, cifar10_classes).flatten()]
    cifar10_y_test_filtered = cifar10_y_test[np.isin(cifar10_y_test, cifar10_classes).flatten()]
    
    return cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_x_test_filtered, cifar10_y_test_filtered

def filter_cifar100(cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test):
    cifar100_x_train_filtered = cifar100_x_train[np.isin(cifar100_y_train, cifar100_classes).flatten()]
    cifar100_y_train_filtered = cifar100_y_train[np.isin(cifar100_y_train, cifar100_classes).flatten()]
    cifar100_x_test_filtered = cifar100_x_test[np.isin(cifar100_y_test, cifar100_classes).flatten()]
    cifar100_y_test_filtered = cifar100_y_test[np.isin(cifar100_y_test, cifar100_classes).flatten()]
    
    return cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_x_test_filtered, cifar100_y_test_filtered

# Filter CIFAR10 and CIFAR100 data
cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_x_test_filtered, cifar10_y_test_filtered = filter_cifar10(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test)
cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_x_test_filtered, cifar100_y_test_filtered = filter_cifar100(cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test)

# Printing the shape of the filtered CIFAR10 and CIFAR100 data
print("Filtered CIFAR10 X Train Shape: ", cifar10_x_train_filtered.shape, ", Y Train Shape: ", cifar10_y_train_filtered.shape)
print("Filtered CIFAR10 X Test Shape: ", cifar10_x_test_filtered.shape,  ", Y Test Shape: ", cifar10_y_test_filtered.shape)
print("Filtered CIFAR100 X Train Shape: ", cifar100_x_train_filtered.shape,  ", Y Train Shape: ", cifar100_y_train_filtered.shape)
print("Filtered CIFAR100 X Test Shape: ", cifar100_x_test_filtered.shape,  ", Y Test Shape: ", cifar100_y_test_filtered.shape)
