import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluates the performance of a regression model and provides a summary.

    Parameters:
        model: A trained regression model from scikit-learn.
        X_test: Test features.
        y_test: Test labels.
    """
    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)

    # Display results
    print(f"Classification Model {model} Evaluation:")
    print("===========================")
    print(f"F1 score: {f1:.2f}")


train_data = np.genfromtxt('./data/train.csv', delimiter=';')
val_data = np.genfromtxt('./data/val.csv', delimiter=';')
test_data = np.genfromtxt('./data/test.csv', delimiter=';')

X_train = train_data[:, :-1]
y_train = train_data[:, -1].astype(int)
X_val = val_data[:, :-1]
y_val = val_data[:, -1].astype(int)
X_test = test_data[:, :-1]
y_test = test_data[:, -1].astype(int)

rfc = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=50)
scaler = StandardScaler()

rfc_pipe = Pipeline([
    ('scaler', scaler),
    ('model', rfc),
])

rfc_pipe.fit(X_train, y_train).score(X_val, y_val)
evaluate_classification_model(rfc_pipe, X_test, y_test)


