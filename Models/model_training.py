from sklearn.tree import DecisionTreeClassifier


def train_model(X_train, y_train, random_state=42):
    """Train Decision Tree model."""

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    return model