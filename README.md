# Iris Flower Classification
A machine learning project that classifies Iris flowers into three species that is Setosa, Versicolor and Virginica using a Decision Tress Classifier

## Features
Iris_Flower_Classification/

├── main.py                  # Entry point - runs the full pipeline

├── config.py                # Configuration settings (paths, parameters)

├── requirements.txt         # Required Python packages

│

├── Data/

│   ├── data_loader.py       # Loads the dataset from CSV

│   └── Iris.csv             # Iris dataset

│

├── Preprocessing/

│   └── preprocessing.py     # Feature/label splitting and train-test split

│

├── Models/

│   └── model_training.py    # Decision Tree model training

│

├── Evaluation/

│   └── evaluation.py        # Accuracy, classification report, confusion matrix

│

├── Visualization/

│   └── visualization.py     # Feature plot and confusion matrix heatmap

│

├── Outputs/

│   ├── feature_plot/        # Saved feature scatter plot

│   └── confusion_matrix/    # Saved confusion matrix heatmap

│

└── Tests/

├── test_data_loader.py

├── test_model.py

└── test_preprocessing.py

## Setup Instructions

1. Clone the repository:
   git clone https://github.com/VineshSri/Iris_Flower_Classification.git
   cd Iris_Flower_Classification

2. Install dependencies:
   pip install -r requirements.txt

3. Run the project:
   python main.py

## Output

After running, the following files will be generated:
- `Outputs/feature_plot/feature_plot.png` — Scatter plot of petal features colored by species
- `Outputs/confusion_matrix/confusion_matrix.png` — Heatmap of model predictions

## Dataset

The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Model

- **Algorithm:** Decision Tree Classifier
- **Train/Test Split:** 80% / 20%
- **Accuracy:** 100% on test set

## Dependencies

- pandas
- scikit-learn
- matplotlib
- seaborn
- pytest
