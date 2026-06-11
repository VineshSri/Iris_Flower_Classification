from config import TEST_SIZE, RANDOM_STATE, DATA_PATH
from config import CONFUSION_MATRIX_PATH, FEATURE_PLOT_PATH

from Data.data_loader import load_data

from Preprocessing.preprocessing import preprocess_data
from Preprocessing.preprocessing import split_dataset

from Models.model_training import train_model

from Evaluation.evaluation import evaluate_model

from Visualization.visualization import save_feature_plot, save_confusion_matrix


def main():
    # --- Load ---
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    # --- Preprocess ---
    print("Preprocessing dataset...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE, RANDOM_STATE)

    # --- Train ---
    print("Training model...")
    model = train_model(X_train, y_train, random_state=RANDOM_STATE)

    # --- Evaluate ---
    print("Evaluating model...")
    accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("\nConfusion Matrix:\n")
    print(matrix)

    # --- Visualize ---
    print("\nSaving visualizations...")
    labels = sorted(y.unique())
    save_feature_plot(df, FEATURE_PLOT_PATH)
    save_confusion_matrix(matrix, labels, CONFUSION_MATRIX_PATH)

    print("\nDone! All outputs saved.")


if __name__ == "__main__":
    main()
    
