import numpy as np
from keras.datasets import cifar10, cifar100

# Load CIFAR10 and CIFAR100 data
(cifar10_train_img, cifar10_train_label), (cifar10_test_img, cifar10_test_label) = cifar10.load_data()
(cifar100_train_img, cifar100_train_label), (cifar100_test_img, cifar100_test_label) = cifar100.load_data()

# Select required data from CIFAR10 and CIFAR100
cifar10_classes = [1, 2, 3, 5, 7, 9]
cifar100_classes = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]