import numpy as np
import matplotlib.pyplot as plt
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

# Plotting CIFAR10 and CIFAR100 data
def plot_cifar(x_train, y_train, num_of_img):
    fig, axes = plt.subplots(1, num_of_img, figsize=(10, 10))
    for i in range(num_of_img):
        # Choose random images from the training set
        index = np.random.randint(0, x_train.shape[0])
        axes[i].set_title("Class: " + str(y_train[index][0]))
        #Turn off the axis for better visualization
        axes[i].axis('off')
        axes[i].imshow(x_train[index])
    plt.show()

# Plotting filtered CIFAR10 and CIFAR100 data
def plot_filtered_cifar(x, y, class_labels, num_images):
    np.random.seed(0)
    # Get the number of classes
    num_classes = len(class_labels)
    # Create a subplot with a grid of size (num_classes, num_images)
    fig, axes = plt.subplots(num_classes, num_images, figsize=(2 * num_images, 2 * num_classes))
    # Adjust the layout for better spacing
    plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=0.5)
    
    # Loop through each class
    for i, class_label in enumerate(class_labels):
        # Find indices where the label matches the current class
        indices = np.where(y == class_label)[0]
        # Choose random images for each class
        random_indices = np.random.choice(indices, num_images, replace=False)
        
        # Loop through each image in the current class
        for j, idx in enumerate(random_indices):
            # Display the image on the subplot
            axes[i][j].imshow(x[idx])
            # Turn off the axis for better visualization
            axes[i][j].axis('off')
            axes[i][j].set_title(f'Class: {class_label}' if j == 0 else '', size='large')
    plt.show()

# Combine CIFAR-10 and CIFAR-100 data
def combine_cifar(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test, cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test):
    x_train = np.concatenate((cifar10_x_train, cifar100_x_train), axis=0)
    y_train = np.concatenate((cifar10_y_train, cifar100_y_train), axis=0)
    x_test = np.concatenate((cifar10_x_test, cifar100_x_test), axis=0)
    y_test = np.concatenate((cifar10_y_test, cifar100_y_test), axis=0)
    
    return x_train, y_train, x_test, y_test

# Filter CIFAR10 and CIFAR100 data
cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_x_test_filtered, cifar10_y_test_filtered = filter_cifar10(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test)
cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_x_test_filtered, cifar100_y_test_filtered = filter_cifar100(cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test)


# Display shapes of CIFAR-10 training and testing data
print(f"CIFAR-10 Training Data: X Shape - {cifar10_x_train.shape}, Y Shape - {cifar10_y_train.shape}")
print(f"CIFAR-10 Testing Data: X Shape - {cifar10_x_test.shape}, Y Shape - {cifar10_y_test.shape}")

# Display shapes of CIFAR-100 training and testing data
print(f"CIFAR-100 Training Data: X Shape - {cifar100_x_train.shape}, Y Shape - {cifar100_y_train.shape}")
print(f"CIFAR-100 Testing Data: X Shape - {cifar100_x_test.shape}, Y Shape - {cifar100_y_test.shape}")


# Display shapes of filtered CIFAR-10 training and testing data
print(f"\nFiltered CIFAR-10 Training Data: X Shape - {cifar10_x_train_filtered.shape}, Y Shape - {cifar10_y_train_filtered.shape}")
print(f"Filtered CIFAR-10 Testing Data: X Shape - {cifar10_x_test_filtered.shape}, Y Shape - {cifar10_y_test_filtered.shape}")

# Display shapes of filtered CIFAR-100 training and testing data
print(f"Filtered CIFAR-100 Training Data: X Shape - {cifar100_x_train_filtered.shape}, Y Shape - {cifar100_y_train_filtered.shape}")
print(f"Filtered CIFAR-100 Testing Data: X Shape - {cifar100_x_test_filtered.shape}, Y Shape - {cifar100_y_test_filtered.shape}")


# Plot images from CIFAR-10
plot_cifar(cifar10_x_train, cifar10_y_train, 5)
# Plot images from CIFAR-100
plot_cifar(cifar100_x_train, cifar100_y_train, 5)

# Plot filtered images from CIFAR-10
plot_filtered_cifar(cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_classes, 5)
# Plot filtered images from CIFAR-100
plot_filtered_cifar(cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_classes, 5)


# Combine CIFAR-10 and CIFAR-100 data using the combine_cifar function
x_train, y_train, x_test, y_test = combine_cifar(
    cifar10_x_train_filtered, cifar10_y_train_filtered,
    cifar10_x_test_filtered, cifar10_y_test_filtered,
    cifar100_x_train_filtered, cifar100_y_train_filtered,
    cifar100_x_test_filtered, cifar100_y_test_filtered
)

# Get the unique classes in the combined dataset
combined_classes = np.unique(np.concatenate((y_train, y_test)))

# Display the shape of the combined dataset
print("Combined Train Shape:", x_train.shape, y_train.shape)
print("Combined Test Shape:", x_test.shape, y_test.shape)

# Display the unique combined classes
print("Combined Classes:", combined_classes)
