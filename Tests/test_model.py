from data.data_loader import load_data
from preprocessing.preprocessing import preprocess_data
from preprocessing.preprocessing import split_dataset
from models.model_training import train_model


def test_model_training():
    df = load_data("data/Iris.csv")

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        0.2,
        42
    )

    model = train_model(X_train, y_train)

    assert model is not None
