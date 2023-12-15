import numpy as np
from keras.datasets import cifar10, cifar100

# Load CIFAR10 and CIFAR100 data
(cifar10_train_img, cifar10_train_label), (cifar10_test_img, cifar10_test_label) = cifar10.load_data()
(cifar100_train_img, cifar100_train_label), (cifar100_test_img, cifar100_test_label) = cifar100.load_data(label_mode='fine')

