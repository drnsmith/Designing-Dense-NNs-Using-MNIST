import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/dense_mnist_model.h5"

def evaluate_model():
    """Evaluate the trained model on test data."""
    # Load test data
    test_images = np.load(os.path.join(PROCESSED_DATA_PATH, "test_images.npy"))
    test_labels = np.load(os.path.join(PROCESSED_DATA_PATH, "test_labels.npy"))

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Evaluate and print metrics
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    print(classification_report(test_labels, predicted_classes))

if __name__ == "__main__":
    evaluate_model()
