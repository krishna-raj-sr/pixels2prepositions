import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
import constants as c


def get_model(input_shape=c.IMAGE_FULL_DIMENSION):
    # Clear any existing TensorFlow graph
    tf.keras.backend.clear_session()
    
    # Initialize a sequential model
    model = Sequential()

    # Add the first convolutional layer
    model.add(Conv2D(input_shape=input_shape, filters=6, kernel_size=(3, 3), padding="same", activation="relu"))
    # Add max pooling layer
    model.add(MaxPool2D(pool_size=(3, 3), strides=(1, 1)))

    # Add the second convolutional layer
    model.add(Conv2D(filters=3, kernel_size=(5, 5), padding="same", activation="relu"))
    # Add max pooling layer
    model.add(MaxPool2D(pool_size=(5, 5), strides=(2, 2)))

    # Add the third convolutional layer
    model.add(Conv2D(filters=3, kernel_size=(5, 5), padding="same", activation="relu"))
    # Add max pooling layer
    model.add(MaxPool2D(pool_size=(5, 5), strides=(2, 2)))

    # Flatten the output from convolutional layers
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(units=5, activation="relu"))
    model.add(Dense(units=5, activation="relu"))

    # Output layer with softmax activation for classification
    model.add(Dense(units=2, activation="softmax"))

    return model
