import os
import numpy as np
from tensorflow.keras.datasets import mnist

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

def load_and_preprocess_mnist():
    """Load and preprocess the MNIST dataset."""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Save processed data
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_PATH, "train_images.npy"), train_images)
    np.save(os.path.join(PROCESSED_DATA_PATH, "train_labels.npy"), train_labels)
    np.save(os.path.join(PROCESSED_DATA_PATH, "test_images.npy"), test_images)
    np.save(os.path.join(PROCESSED_DATA_PATH, "test_labels.npy"), test_labels)

if __name__ == "__main__":
    load_and_preprocess_mnist()
