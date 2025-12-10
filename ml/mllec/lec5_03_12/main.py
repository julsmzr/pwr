import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from random_subspace import RandomSubspace

def load_dataset():
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=0, n_repeated=0, n_informative=10, random_state=1410, class_sep=0.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, y_train, X_test, y_test

def predict(clf, X_train, y_train, X_test, y_test):
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test, preds)
    return score

def perform_rskf(clf, X_train, y_train, X_test, y_test):
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
    scores =np.zeros((10))

    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scores[i] = predict(clf, X_train, y_train, X_test, y_test)

    return np.mean(scores)


if __name__ == "__main__":

    X, y, X_train, y_train, X_test, y_test = load_dataset()
        
    print("-" * 50)

    print("    XGB    ", round(perform_rskf(XGBClassifier(n_estimators=10), X_train, y_train, X_test, y_test)), 3)
    
    print("-" * 50)

    print("    GNB    ", round(perform_rskf(GaussianNB(), X_train, y_train, X_test, y_test)), 3)
    print("RS GNB hard", round(perform_rskf(RandomSubspace(n_features=5, n_estimators=100, random_state=None, voting="hard"), X_train, y_train, X_test, y_test)), 3)
    print("RS GNB soft", round(perform_rskf(RandomSubspace(n_features=5, n_estimators=100, random_state=None, voting="soft"), X_train, y_train, X_test, y_test)), 3)
    
    print("-" * 50)
    
    print("    DT     ", round(perform_rskf(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)), 3)
    print("RS DT hard ", round(perform_rskf(RandomSubspace(base_estimator=DecisionTreeClassifier(), n_features=5, n_estimators=100, random_state=None, voting="hard"), X_train, y_train, X_test, y_test)), 3)
    print("RS DT soft ", round(perform_rskf(RandomSubspace(base_estimator=DecisionTreeClassifier(), n_features=5, n_estimators=100, random_state=None, voting="soft"), X_train, y_train, X_test, y_test)), 3)

    print("-" * 50)