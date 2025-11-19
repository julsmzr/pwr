import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy.spatial.distance import cdist


def task1():
    print("\nTask 1")
    print("-" * 50)

    X, y = make_classification(weights=[0.7, 0.3])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    class MajorityClassifier(BaseEstimator, ClassifierMixin):

        def __init__(self):
            pass

        def fit(self, X, y):
            label, count = np.unique(y, return_counts=True)
            self.majority_label = label[np.argmax(count)]
            return self

        def predict(self, X):
            return np.full(len(X), self.majority_label)

    clf = MajorityClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bac = balanced_accuracy_score(y_test, y_pred)

    print("Accuracy score:", round(acc, 3))
    print("Balanced Accuracy score:", round(bac, 3))

    print("-" * 50)

def task2():
    print("\nTask 2")
    print("-" * 50)

    X, y = make_classification(weights=[0.7, 0.3])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    class CentroidClassifier(BaseEstimator, ClassifierMixin):

        def __init__(self):
            pass

        def fit(self, X, y):
            self.centroids = np.asarray(
                [np.mean(X[y==label], axis=0) for label in np.unique(y)]
                )
            return self

        def predict(self, X):
            return np.argmin(cdist(X, self.centroids), axis=1)

    clf = CentroidClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bac = balanced_accuracy_score(y_test, y_pred)

    print("Accuracy score:", round(acc, 3))
    print("Balanced Accuracy score:", round(bac, 3))

    print("-" * 50)

def task3():
    
    print("\nTask 3")
    print("-" * 50)

    class CentroidClassifier(BaseEstimator, ClassifierMixin):

        def __init__(self):
            pass

        def fit(self, X, y):
            self.centroids = np.asarray(
                [np.mean(X[y==label], axis=0) for label in np.unique(y)]
                )
            return self

        def predict(self, X):
            return np.argmin(cdist(X, self.centroids), axis=1)

    datasets = [
        make_classification(n_informative=10, n_clusters_per_class=1, n_classes=2, weights=[0.9, 0.1]),
        make_classification(n_informative=10, n_clusters_per_class=1, n_classes=3),
        make_classification(n_informative=10, n_clusters_per_class=1, n_classes=5, class_sep=0.5, weights=[0.1, 0.2, 0.3, 0.3, 0.1]),
    ]
    n_datasets = len(datasets)

    clfs = {
        "CentroidClassifier": CentroidClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier()
    }
    n_clfs = len(clfs)

    n_splits = 2
    n_repeats = 5

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    # DATASETS x FOLDS x CLFS
    scores = np.full((n_datasets, n_splits * n_repeats, n_clfs), np.nan)

    for dataset_idx, (X, y) in enumerate(datasets):
        for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            for clf_idx, clf in enumerate(clfs.values()):
                clf = clone(clf)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                score = balanced_accuracy_score(y_test, y_pred)
                scores[dataset_idx, fold_idx, clf_idx] = score  
    
    # DATASETS x CLFS
    means = np.mean(scores, axis=1)
    stds = np.std(scores, axis=1)

    for dataset_idx, (X, y) in enumerate(datasets):
        print("Dataset %s" % dataset_idx)
        for clf_idx, clf in enumerate(clfs.keys()):
            print("%s: %s (%s)" % (clf, round(means[dataset_idx, clf_idx], 3), round(stds[dataset_idx, clf_idx], 3)))
        print()

    print("-" * 50)

if __name__ == "__main__":
    task1()
    task2()
    task3()
