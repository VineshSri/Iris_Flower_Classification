from data.data_loader import load_data
from preprocessing.preprocessing import preprocess_data


def test_preprocess_data():
    df = load_data("data/Iris.csv")

    X, y = preprocess_data(df)

    assert X is not None
    assert y is not None
