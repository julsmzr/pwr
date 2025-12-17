import warnings

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.base import clone


def task1():
    print("\nTask 1")
    print("-" * 50)

    X, y = load_digits(return_X_y=True)

    print(X.shape, y.shape)
    print()
    print(X[0])
    print()
    print(np.reshape(X[0], shape=(8, 8)))

    clfs = {
        "GNB": GaussianNB(),
        "MLP": MLPClassifier(),
        "DT": DecisionTreeClassifier()
    }

    n_clfs = len(clfs)
    n_repeats = 5
    n_splits = 2

    rkf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits)
    scores = np.full(shape=(n_clfs, n_splits * n_repeats), fill_value=np.nan)

    for clf_idx, clf in enumerate(clfs.values()):
        for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_idx, fold_idx] = accuracy_score(y_test, y_pred) 
    
    print("\nOriginal digits dataset:")
    for clf_idx, clf_name in enumerate(clfs.keys()):
        print("%s\t %s (%s)" % (clf_name, round(np.mean(scores[clf_idx]), 3), round(np.std(scores[clf_idx]), 3)))

    print("-" * 50)

def task2():
    print("\nTask 2")
    print("-" * 50)

    X, y = load_digits(return_X_y=True)

    clfs = {
        "GNB": GaussianNB(),
        "MLP": MLPClassifier(),
        "DT": DecisionTreeClassifier()
    }

    n_clfs = len(clfs)
    n_repeats = 5
    n_splits = 2

    rkf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits)
    scores = np.full(shape=(n_clfs, n_splits * n_repeats), fill_value=np.nan)

    print_sample = True
    for clf_idx, clf in enumerate(clfs.values()):
        for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            if print_sample:
                print(X_train)
                print_sample = False

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_idx, fold_idx] = accuracy_score(y_test, y_pred) 
    
    print("\nStandard scaler:")
    for clf_idx, clf_name in enumerate(clfs.keys()):
        print("%s\t %s (%s)" % (clf_name, round(np.mean(scores[clf_idx]), 3), round(np.std(scores[clf_idx]), 3)))

    print("-" * 50)

def task3():

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("\nTask 3")
    print("-" * 50)

    X, y = load_digits(return_X_y=True)

    methods = [
        "PCA",
        "SelectKBest"
    ]

    clfs = {
        "GNB": GaussianNB(),
        "MLP": MLPClassifier(),
        "DT": DecisionTreeClassifier()
    }

    n_methods = len(methods)
    n_clfs = len(clfs)
    n_repeats = 5
    n_splits = 2

    rkf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits)
    scores = np.full(shape=(n_methods, n_clfs, n_splits * n_repeats), fill_value=np.nan)

    print_sample = 2
    for clf_idx, clf in enumerate(clfs.values()):
        for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
            # PCA
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            pca = PCA(n_components=8)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            if print_sample == 2:
                print("Instance after PCA:")
                print(X_train[0])
                print_sample -= 1

            clf = clone(clf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[0, clf_idx, fold_idx] = accuracy_score(y_test, y_pred) 

            # SelectKBest
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            select_k_best = SelectKBest(k=8)
            X_train = select_k_best.fit_transform(X_train, y_train)
            X_test = select_k_best.transform(X_test)

            if print_sample == 1:
                print("\nInstance after SelectKBest:")
                print(X_train[0])
                print_sample -= 1

            clf = clone(clf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[1, clf_idx, fold_idx] = accuracy_score(y_test, y_pred) 
    
    print("\nPCA:")
    for clf_idx, clf_name in enumerate(clfs.keys()):
        print("%s\t %s (%s)" % (clf_name, round(np.mean(scores[0, clf_idx]), 3), round(np.std(scores[0, clf_idx]), 3)))

    print("\nSelectKBest:")
    for clf_idx, clf_name in enumerate(clfs.keys()):
        print("%s\t %s (%s)" % (clf_name, round(np.mean(scores[1, clf_idx]), 3), round(np.std(scores[1, clf_idx]), 3)))

    print("-" * 50)

if __name__ == "__main__":
    task1()
    task2()
    task3()
