import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/dense_mnist_model.h5"

def build_dense_model():
    """Build a dense neural network model."""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.01), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model():
    """Train the dense neural network model."""
    # Load preprocessed data
    train_images = np.load(os.path.join(PROCESSED_DATA_PATH, "train_images.npy"))
    train_labels = np.load(os.path.join(PROCESSED_DATA_PATH, "train_labels.npy"))
    test_images = np.load(os.path.join(PROCESSED_DATA_PATH, "test_images.npy"))
    test_labels = np.load(os.path.join(PROCESSED_DATA_PATH, "test_labels.npy"))

    model = build_dense_model()
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

if __name__ == "__main__":
    train_model()
