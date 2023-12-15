import numpy as np
from keras.datasets import cifar10, cifar100

# Load CIFAR10 and CIFAR100 data
(cifar10_x_train, cifar10_y_train), (cifar10_x_train, cifar10_y_train) = cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_train, cifar100_y_train) = cifar100.load_data()

# Select required data from CIFAR10 and CIFAR100
cifar10_classes = [1, 2, 3, 5, 7, 9]
cifar100_classes = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]

def filter_cifar10()