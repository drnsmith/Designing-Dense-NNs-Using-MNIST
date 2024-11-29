### Designing Dense Neural Networks Using MNIST

This repository contains a project for building, training, and evaluating dense neural networks using the MNIST and Fashion-MNIST datasets. It demonstrates the design of efficient architectures and practical insights into hyperparameter tuning, activation functions, and performance evaluation.

#### Project Structure
- `data/`: Contains raw and pre-processed dataset files.
- `notebooks/`: Jupyter Notebooks for exploratory analysis and prototyping.
- `models/`: Stores saved model checkpoints.
- `src/`: Source code for preprocessing, training, and evaluation modules.
- `outputs/`: Stores logs, plots, and evaluation metrics.

#### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Designing-Dense-NNs-Using-MNIST.git

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Dataset
Download the MNIST or Fashion-MNIST dataset using TensorFlow or PyTorch. The dataset is loaded automatically using the scripts provided.

4. Usage
 - Pre-process the Data:

```bash
python src/data_preprocessing.py
```
 - Train a Model:
```bash
python src/model_training.py
```
 - Evaluate a Model:
```bash
python src/model_evaluation.py
```
 - View Results: Outputs will be saved in the `outputs/` folder.

#### Structure
```bash
Designing-Dense-NNs-Using-MNIST/
├── data/
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Preprocessed dataset files
├── notebooks/                 # Jupyter Notebooks for experiments
├── models/                    # Saved model checkpoints
├── src/                       # Source code for project modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── utils.py
├── outputs/                   # Results like plots, logs, and metrics
│   ├── logs/
│   ├── visualizations/
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Files and folders to ignore in Git
├── LICENSE                    # License for the project
└── setup.py                   # Package setup (optional)
```

### License
MIT License

