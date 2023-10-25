import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    images = []
    labels = []

    # Iterate through each directory in data_dir
    for category in range(NUM_CATEGORIES):
        # Join the path of the current directory
        path = os.path.join(data_dir, str(category))
        for file in os.listdir(path):
            # Read the image using OpenCV
            img = cv2.imread(os.path.join(path, file))
            # Resize the image to the desired dimensions
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # Append the image and its corresponding label to images and labels lists
            images.append(img)
            labels.append(category)
    
    return (images, labels)

def get_model():
    model = tf.keras.models.Sequential([
        
        # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Max-pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten layer to convert 3D data to 1D
        tf.keras.layers.Flatten(),
        
        # Hidden layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Output layer with NUM_CATEGORIES units and softmax activation
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    main()
