from sklearn.model_selection import train_test_split


def preprocess_data(df):
    """Split dataset into features and labels."""

    X = df.drop(["Species", "Id"], axis=1)
    y = df["Species"]

    return X, y


def split_dataset(X, y, test_size, random_state):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
