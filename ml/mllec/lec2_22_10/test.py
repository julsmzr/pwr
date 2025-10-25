# Showcase proper 1-NN classifcation and Cross Validation Evaluation

import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

from onenn import OneNN

def load_dataset():
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_repeated=0, n_informative=2, random_state=188, class_sep=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, y_train, X_test, y_test

def predict(clf, X_train, y_train, X_test, y_test):
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    score=accuracy_score(y_test, preds)
    return score

def perform_rskf(clf, X_train, y_train, X_test, y_test):

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
    scores =np.zeros((2, 10))

    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scores[0, i] = predict(OneNN(), X_train, y_train, X_test, y_test)
        scores[1, i] = predict(LogisticRegression(), X_train, y_train, X_test, y_test)

    return scores


if __name__ == "__main__":

    # Load and split synthetic data
    X, y, X_train, y_train, X_test, y_test = load_dataset()

    # Print regular prediction
    print("Prediction:", predict(OneNN(), X_train, y_train, X_test, y_test))

    # Sanity check
    assert 1.0 == predict(OneNN(), X, y, X, y)

    scores = perform_rskf(OneNN(), X_train, y_train, X_test, y_test)
    # scores = perform_rskf(LogisticRegression(), X_train, y_train, X_test, y_test)

    # Save results for later analysis
    np.save("data/experiment_results.npy", scores)
