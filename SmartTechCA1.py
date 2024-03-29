import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# Load CIFAR10 and CIFAR100 data
(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()

# Select required data from CIFAR10 and CIFAR100
cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
cifar100_classes = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]

# Function for filtering CIFAR10 and CIFA100 data
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

# Plotting CIFAR10 and CIFAR100 data (Data Exploration)
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
def plot_filtered_cifar(x, y, class_labels, num_of_img):
    np.random.seed(0)
    # Get the number of classes
    num_of_class = len(class_labels)
    # Create a subplot with a grid of size (num_classes, num_images)
    fig, axes = plt.subplots(num_of_class, num_of_img, figsize=(2 * num_of_img, 2 * num_of_class))
    # Adjust the layout for better spacing
    plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=0.5)
    
    # Loop through each class
    for i, class_label in enumerate(class_labels):
        # Find indices where the label matches the current class
        indices = np.where(y == class_label)[0]
        # Choose random images for each class
        random_indices = np.random.choice(indices, num_of_img, replace=False)
        
        # Loop through each image in the current class
        for j, idx in enumerate(random_indices):
            # Display the image on the subplot
            axes[i][j].imshow(x[idx])
            # Turn off the axis for better visualization
            axes[i][j].axis('off')
            axes[i][j].set_title(f'Class: {class_label}' if j == 0 else '', size='large')
    plt.show()

# Combine CIFAR-10 and CIFAR100 data
def combine_cifar(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test, cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test):
    x_train = np.concatenate((cifar10_x_train, cifar100_x_train), axis=0)
    y_train = np.concatenate((cifar10_y_train, cifar100_y_train), axis=0)
    x_test = np.concatenate((cifar10_x_test, cifar100_x_test), axis=0)
    y_test = np.concatenate((cifar10_y_test, cifar100_y_test), axis=0)
    
    return x_train, y_train, x_test, y_test

# Display combined CIFAR10 and CIFAR100 data
def display_combined_cifar(x, y, class_labels, num_of_img=5):
    num_classes = len(class_labels)
    cols = num_of_img
    
    # Create a subplot grid with dimensions (num_classes, num_of_img)
    fig, axes = plt.subplots(num_classes, num_of_img, figsize=(2 * num_of_img, 2 * num_classes))
    
    # Adjust the layout for better spacing
    plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=0.5)
    num_of_data = []
    
    # If there's only one class, convert axes to a 2D array
    if num_classes == 1:
        axes = np.array([axes])
    for i in range(cols):
        for j, class_label in enumerate(class_labels):
            # Find indices where the label matches the current class
            indices = np.where(y.flatten() == class_label)[0]
            
            # Randomly select num_of_img indices without replacement
            random_indices = np.random.choice(indices, num_of_img, replace=False)
            
            # Loop through each image in the current class
            for k, idx in enumerate(random_indices):
                # Display the image on the subplot
                axes[j, i].imshow(x[idx], interpolation='nearest')
                axes[j, i].axis('off')
                
                # Set the title for the last column of subplots
                if i == cols - 1: 
                    num_of_data.append(len(indices))
                    axes[j, i].set_title(f'Class: {class_label}', size='large')
                    
    # Adjust the spacing of the subplots
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.show()
    return num_of_data

# Apply grayscale filter
def grayscale_filter(img):
    # Convert RGB to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

# Apply Equalization filter
def equalization_filter(img):
    # Apply histogram equalization
    img = cv2.equalizeHist(img)
    
    return img

# Apply Gaussian filter
def gaussian_filter(img):
    # Apply Gaussian filter
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

# Scale down
def scale_down_variance(img):
    return img / 255

# Apply all filters
def preprocess(img):
    # Apply grayscale filter
    img = grayscale_filter(img)
    # Apply equalization filter
    img = equalization_filter(img)
    # Apply Gaussian filter
    img = gaussian_filter(img)
    # Normalize pixel values to the range [0, 1]
    # Helps in stabilizing the learning process and can lead to faster convergence during training
    img = scale_down_variance(img)
    
    return img

# Reshape data
def reshape(images):
    return images.reshape(images.shape[0], 32, 32, 1)
    
# Build the model
def leNet_model():
  model = Sequential()
  # Convolutional layer with 60 filters, kernel size (5, 5), and ReLU activation
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  # Max pooling layer with pool size (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Convolutional layer with 30 filters, kernel size (3, 3), and ReLU activation
  model.add(Conv2D(30, (3, 3), activation='relu'))
  # Max pooling layer with pool size (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Flatten the output for dense layers
  model.add(Flatten())
  # Dense layer with 500 neurons and ReLU activation
  model.add(Dense(500, activation='relu'))
  # Dropout layer with a dropout rate of 0.5
  model.add(Dropout(0.5))
  # Output layer with 'num_classes' neurons and softmax activation
  model.add(Dense(num_of_data, activation='softmax'))
  # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
  model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# Build modified model
def modified_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Increasing the depth and adding more layers
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Global Average Pooling
  GlobalAveragePooling2D(),  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_of_data, activation='softmax'))
  model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def evaluate_model(model, x_test, y_test):
    # Evaluate the model on the test set
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

def analyze_model(history):
    # Plot the training accuracy and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history):
    # Plot the training loss and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# Display the size of the first image in CIFAR-10
print("CIFAR-10 Image Size:", cifar10_x_train[0].shape)

# Display the size of the first image in CIFAR-100
print("CIFAR-100 Image Size:", cifar100_x_train[0].shape)



# Count the number of labels for each image in CIFAR-10
cifar10_label_counts = np.sum(cifar10_y_train, axis=1)
print("CIFAR-10 Label Counts:", cifar10_label_counts)

# Count the number of labels for each image in CIFAR-100
cifar100_label_counts = np.sum(cifar100_y_train, axis=1)
print("CIFAR-100 Label Counts:", cifar100_label_counts)


# Unique classes in CIFAR-10
print("CIFAR-10 Unique Classes:", np.unique(cifar10_y_train))

# Unique classes in CIFAR-100
print("CIFAR-100 Unique Classes:", np.unique(cifar100_y_train))


# Filter CIFAR10 and CIFAR100 data
cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_x_test_filtered, cifar10_y_test_filtered = filter_cifar10(cifar10_x_train, cifar10_y_train, cifar10_x_test, cifar10_y_test)
cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_x_test_filtered, cifar100_y_test_filtered = filter_cifar100(cifar100_x_train, cifar100_y_train, cifar100_x_test, cifar100_y_test)


# Display shapes of CIFAR10 training and testing data (Data Exploration)
print(f"\nCIFAR-10 Training Data: X Shape - {cifar10_x_train.shape}, Y Shape - {cifar10_y_train.shape}")
print(f"CIFAR-10 Testing Data: X Shape - {cifar10_x_test.shape}, Y Shape - {cifar10_y_test.shape}")

# Display shapes of CIFAR100 training and testing data
print(f"CIFAR-100 Training Data: X Shape - {cifar100_x_train.shape}, Y Shape - {cifar100_y_train.shape}")
print(f"CIFAR-100 Testing Data: X Shape - {cifar100_x_test.shape}, Y Shape - {cifar100_y_test.shape}")


# Display shapes of filtered CIFAR10 training and testing data (Data Exploration)
print(f"\nFiltered CIFAR-10 Training Data: X Shape - {cifar10_x_train_filtered.shape}, Y Shape - {cifar10_y_train_filtered.shape}")
print(f"Filtered CIFAR-10 Testing Data: X Shape - {cifar10_x_test_filtered.shape}, Y Shape - {cifar10_y_test_filtered.shape}")

# Display shapes of filtered CIFAR100 training and testing data
print(f"Filtered CIFAR-100 Training Data: X Shape - {cifar100_x_train_filtered.shape}, Y Shape - {cifar100_y_train_filtered.shape}")
print(f"Filtered CIFAR-100 Testing Data: X Shape - {cifar100_x_test_filtered.shape}, Y Shape - {cifar100_y_test_filtered.shape}")


# Plot images from CIFAR10
plot_cifar(cifar10_x_train, cifar10_y_train, 5)
# Plot images from CIFAR100
plot_cifar(cifar100_x_train, cifar100_y_train, 5)

# Plot filtered images from CIFAR10
plot_filtered_cifar(cifar10_x_train_filtered, cifar10_y_train_filtered, cifar10_classes, 5)
# Plot filtered images from CIFAR100
plot_filtered_cifar(cifar100_x_train_filtered, cifar100_y_train_filtered, cifar100_classes, 5)


# Combine CIFAR10 and CIFAR100 data using the combine_cifar function
cifar100_y_train_filtered += 10
cifar100_y_test_filtered += 10
x_train, y_train, x_test, y_test = combine_cifar(
    cifar10_x_train_filtered, cifar10_y_train_filtered,
    cifar10_x_test_filtered, cifar10_y_test_filtered,
    cifar100_x_train_filtered, cifar100_y_train_filtered,
    cifar100_x_test_filtered, cifar100_y_test_filtered
)

# Get the unique classes in the combined dataset (Data Exploration)
combined_classes = np.unique(np.concatenate((y_train, y_test)))

# Display the shape of the combined dataset
print("\nCombined Train Shape:", x_train.shape, y_train.shape)
print("Combined Test Shape:", x_test.shape, y_test.shape)

# Display the unique combined classes
print("Combined Classes:", combined_classes)

# Total number of images for training and testing sets
total_train_images = x_train.shape[0]
total_test_images = x_test.shape[0]


# Display images from combined dataset
combined_cifar = display_combined_cifar(x_train, y_train, combined_classes)


# Plot the combined classes (Data Exploration)
print("\nClasses:", combined_cifar)
num_of_data = len(combined_cifar)
plt.figure(figsize=(12, 4))
plt.bar(range(num_of_data), combined_cifar)
plt.title("Distribution of Images Across Combined Classes")
plt.xlabel("Classes")
plt.ylabel("Number of Data")
plt.show()


# Total number of images for each of the 24 classes
num_images_per_class = [np.sum(y_train[:, i]) for i in range(num_of_data)]

# Display the results
print("Total Training Images:", total_train_images)
print("Total Testing Images:", total_test_images)
print("\nNumber of Images for Each Class:")
for i in range(min(len(combined_classes), len(num_images_per_class))):
    print(f"Class {combined_classes[i]}: {num_images_per_class[i]}")
    

# # Plot grayscale image
# img = x_train[0]
# img_gray = grayscale_filter(img)
# plt.imshow(img_gray)
# plt.title("Grayscale Image")
# plt.axis("off")
# plt.show()

# # Displays the image in shades of gray
# plt.imshow(img_gray, cmap='gray')
# plt.title("Grayscale Image with Colormap")
# plt.axis("off")
# plt.show()

# # Plot equalized image
# img_gray_eq = equalization_filter(img_gray)
# plt.imshow(img_gray_eq)
# plt.title("Equalized Image")
# plt.axis("off")
# plt.show()

# # Displays the image in shades of gray
# plt.imshow(img_gray_eq, cmap='gray')
# plt.title("Equalized Image with Colormap")
# plt.axis("off")
# plt.show()

# # Plot gaussian image
# img_gaussian = gaussian_filter(img)
# plt.imshow(img_gaussian)
# plt.title("Gaussian Image")
# plt.axis("off")
# plt.show()

# # Plot preprocessed image
# img_preprocessed = preprocess(img)
# plt.imshow(img_preprocessed)
# plt.title("Preprocessed Image")
# plt.axis("off")
# plt.show()

# # Displays the image in shades of gray
# plt.imshow(img_preprocessed, cmap='gray')
# plt.title("Preprocessed Image with Colormap")
# plt.axis("off")
# plt.show()

# Plot preprocessed image
x_train = np.array(list(map(preprocess, x_train)))
x_test = np.array(list(map(preprocess, x_test)))

# Randomly select an image from the training set
random_index = np.random.choice(len(x_train))
# Display the preprocessed image
plt.imshow(x_train[random_index])
plt.title("Preprocessed Image x_train")
plt.axis("off")


# Reshape data 
x_train = reshape(x_train)
x_test = reshape(x_test)

print("\nX Train shape: ", x_train.shape)
print("X Test shape: ", x_test.shape)

# Create an ImageDataGenerator with specified augmentation parameters
# Randomly shift, zoom, shear and rotate images
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range = 0.1, zoom_range = 0.2, shear_range = 0.1, rotation_range=10)
datagen.fit(x_train)
# Generate batches of augmented data
batches = datagen.flow(x_train, y_train, batch_size = 20)
x_batch, y_batch = next(batches)

# Create subplots to visualize the augmented images
fig, axs = plt.subplots(1, 20, figsize=(20, 5))
fig.tight_layout()
for i in range(20):
  axs[i].imshow(x_batch[i].reshape(32, 32))
  axs[i].axis('off')

# One hot encoding
y_train = to_categorical(y_train, num_of_data)
y_test = to_categorical(y_test, num_of_data)

# Create the LeNet model 
model = leNet_model()
print(model.summary())

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=50), epochs=20 , validation_data=(x_test, y_test))


# # Create the modified model 
# model = modified_model()
# print(model.summary())


# # Train the model
# history = model.fit(datagen.flow(x_train, y_train, batch_size=50), steps_per_epoch=x_train.shape[0]/50, epochs=20, validation_data=(x_test, y_test), verbose=1, shuffle=1)


# Evaluate the model on the test set
evaluate_model(model, x_test, y_test)

# Plot the training accuracy and validation accuracy
analyze_model(history)

# Plot the training loss and validation loss
plot_loss(history)


# Testing model by tweaking hyperparameters
# epoch 5 (Underfitting)
# loss: 2.0014 - accuracy: 0.3958 - val_loss: 1.7752 - val_accuracy: 0.4612 
# epoch 15
# loss: 1.6859 - accuracy: 0.4864 - val_loss: 1.5435 - val_accuracy: 0.5329
# epoch 20 (Best result)
# loss: 1.5198 -  loss: 1.6576 - accuracy: 0.4969 - val_loss: 1.4822 - val_accuracy: 0.5501
# epoch 30
# loss: 1.6562 - accuracy: 0.4949 - val_loss: 1.4517 - val_accuracy: 0.5566
# epoch 50 (Overfitting)
# loss: 0.8819 - accuracy: 0.7050 - val_loss: 1.8727 - val_accuracy: 0.5008 

# epoch 20 conv2d from 60/30 to 140/70
# loss: 1.6176 - accuracy: 0.5056 - val_loss: 1.4454 - val_accuracy: 0.5562



# Predicting image
# Bird
# url = "https://cdn.pixabay.com/photo/2016/12/13/22/25/bird-1905255_1280.jpg"
# Truck
# url = "https://cdn.pixabay.com/photo/2017/06/11/10/46/truck-2391940_1280.jpg"
# Tree
url = "https://cdn.pixabay.com/photo/2018/01/21/19/57/tree-3097419_1280.jpg"
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Convert the PIL image to a NumPy array
img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocess(img)
img = img.reshape(1, 32, 32, 1)
print("Predicted Image: " + str(np.argmax(model.predict(img), axis=-1)))