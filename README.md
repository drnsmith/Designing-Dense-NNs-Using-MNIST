### Building and Training Dense Neural Network for Classifying Images

#### **Overview**
This project demonstrates the implementation of a dense neural network (DNN) for classifying images in the **Fashion MNIST dataset**. The goal is to explore DNN architecture design, activation functions, and regularisation techniques such as dropout, achieving accurate classification of clothing items.

---

#### **Motivation**
The Fashion MNIST dataset serves as a modern benchmark for evaluating neural network architectures. This project:
1. Highlights how dropout improves generalization by reducing overfitting.
2. Compares ReLU activation against traditional Sigmoid.
3. Demonstrates visualization techniques for understanding the dataset and model performance.

---

#### **Key Features**
- **Dataset Handling**:
  - Loads and preprocesses the Fashion MNIST dataset.
  - Splits the dataset into training and testing subsets.
  - Visualizes sample images to provide insights into class distribution.

- **Model Architecture**:
  - A fully connected neural network with:
    - ReLU activation for non-linearity.
    - Dropout layers (50% rate) for regularization.
  - Categorical Crossentropy as the loss function for classification.

- **Evaluation**:
  - Training and testing accuracy visualized over epochs.
  - Confusion matrix to evaluate per-class predictions.

---

## **Dataset**
- The **Fashion MNIST** dataset is a collection of 70,000 grayscale images of 28x28 pixels across 10 classes.
- Class names: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST.git
cd Designing-Dense-NNs-Using-MNIST
```

### **2. Set Up the Environment**
Create a virtual environment and install the required dependencies:
```bash
python -m venv env
# Activate the virtual environment
# On Windows:
env\Scripts\activate
# On Unix or macOS:
source env/bin/activate
# Install dependencies
pip install -r requirements.txt
```

---

## **Usage**

### **1. Run the Jupyter Notebook**
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook full_code.ipynb
   ```
2. Follow the notebook cells to:
   - Load and preprocess the dataset.
   - Visualise sample images and class distributions.
   - Train the dense neural network.
   - Evaluate performance on the test set.

---

## **Results**
- **Accuracy**: Achieved ~88% test accuracy after training.
- **Loss Curves**: Training and validation loss converged, indicating successful training.
- **Sample Predictions**: The network accurately classified most samples, with occasional confusion between similar classes (e.g., Shirt and T-shirt/top).

---

### Productisation  
This dense neural network (DNN) model for **image classification** can be transformed into a **real-time classification API** that enables businesses and developers to integrate **AI-powered image recognition** into their applications. By optimising the model for **edge deployment on mobile and IoT devices**, it can serve **automated product recognition** for e-commerce platforms, **clothing sorting systems**, or **retail analytics tools**.

### Monetisation  
The project can be monetised through **API-based licensing**, offering **AI-powered image classification** as a service for e-commerce and retail platforms. A **freemium model** could provide **basic image classification for free**, while premium plans could include **custom model training, faster inference, and batch processing**. Additionally, the system could be licensed to **fashion retailers and inventory management solutions** looking to integrate **AI-driven product categorisation**.

---
## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a pull request.

---

## Repository History Cleaned

As part of preparing this repository for collaboration, its commit history has been cleaned. This action ensures a more streamlined project for contributors and removes outdated or redundant information in the history. 

The current state reflects the latest progress as of 24/01/2025.

For questions regarding prior work or additional details, please contact the author.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- **TensorFlow/Keras**: For the deep learning framework.
- **Fashion MNIST Dataset**: A modern replacement for traditional MNIST benchmarks.

