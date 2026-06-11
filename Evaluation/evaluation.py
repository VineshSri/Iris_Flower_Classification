from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix
