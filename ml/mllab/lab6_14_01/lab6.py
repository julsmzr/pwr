import os

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier


from scipy.stats import shapiro, ttest_rel, wilcoxon


def task1():
    print("\nTask 1")
    print("-" * 50)

    data = np.loadtxt("data/task1.csv", delimiter=',', dtype='object')
    X = data[:, :8]
    y = data[:, 8]

    clfs = {
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier()
    }

    n_clfs = len(clfs)
    n_repeats = 5
    n_splits = 2

    scores = np.full(shape=(n_clfs, n_splits * n_repeats), fill_value=np.nan)
    rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits)
   
    for clf_idx, clf in enumerate(clfs.values()):
        for fold_idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_idx, fold_idx] = balanced_accuracy_score(y_test, y_pred) 
    
    alpha = 0.05

    for clf_idx, clf_name in enumerate(clfs.keys()):
        statistic, p_value = shapiro(scores[clf_idx])

        print("Classifier: %s | statistic: %s | p-value: %s" % (clf_name, round(statistic, 3), round(p_value, 3)))
        print("Normal distribution: %s\n" % (p_value > alpha))

    print("-" * 50)

def task2():
    print("\nTask 2")
    print("-" * 50)

    data = np.loadtxt("data/task1.csv", delimiter=',', dtype='object')
    X = data[:, :8]
    y = data[:, 8]

    clfs = {
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier()
    }

    n_clfs = len(clfs)
    n_repeats = 5
    n_splits = 2

    scores = np.full(shape=(n_clfs, n_splits * n_repeats), fill_value=np.nan)
    rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits)
   
    for clf_idx, clf in enumerate(clfs.values()):
        for fold_idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_idx, fold_idx] = balanced_accuracy_score(y_test, y_pred) 
    
    alpha = 0.05

    t_stat = np.full(shape=(n_clfs, n_clfs), fill_value=np.nan)
    p_val = np.full(shape=(n_clfs, n_clfs), fill_value=np.nan)

    better = np.full(shape=(n_clfs, n_clfs), fill_value=np.nan, dtype=bool)
    significance = np.full(shape=(n_clfs, n_clfs), fill_value=np.nan, dtype=bool)

    print_pipeline = []

    for clf_idx_i, clf_name_i in enumerate(clfs.keys()):
        for clf_idx_j, clf_name_j in enumerate(clfs.keys()):
            t_stat[clf_idx_i, clf_idx_j], p_val[clf_idx_i, clf_idx_j] = ttest_rel(scores[clf_idx_i], scores[clf_idx_j])
            
            is_better = t_stat[clf_idx_i, clf_idx_j] > 0
            better[clf_idx_i, clf_idx_j] = is_better

            is_significant = p_val[clf_idx_i, clf_idx_j] < alpha
            significance[clf_idx_i, clf_idx_j] = is_significant

            m1, m2 = round(np.mean(scores[clf_idx_i], axis=0), 3), round(np.mean(scores[clf_idx_j], axis=0), 3)

            if is_significant and is_better:
                print_pipeline.append("%s with mean %f is better than %s with mean %f" % (clf_name_i, m1, clf_name_j, m2))
            elif clf_name_i != clf_name_j:
                print_pipeline.append("No significant difference between %s with mean %f and %s with mean %f" % (clf_name_i, m1, clf_name_j, m2))


    print("t-statistic matrix:\n", t_stat)
    print("\np-value matrix:\n", p_val)
    print()

    print("-" * 5)

    print("\nbetter matrix:\n", better)
    print("\nsignificance matrix:\n", significance)
    print()

    for line in print_pipeline:
        print(line)

    print("-" * 50)

def task3():

    print("\nTask 3")
    print("-" * 50)

    dataset_paths = [os.path.join("data/task3", dataset_filename) for dataset_filename in os.listdir("data/task3")]
    n_datasets = len(dataset_paths)

    data = np.loadtxt("data/task1.csv", delimiter=',', dtype='object')
    X = data[:, :8]
    y = data[:, 8]

    clfs = {
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
    }

    n_clfs = len(clfs)
    n_repeats = 5
    n_splits = 2

    scores = np.full(shape=(n_datasets, n_splits * n_repeats, n_clfs), fill_value=np.nan)
    
   
    for dataset_idx, dataset_filename in enumerate(dataset_paths):
        data = np.loadtxt(dataset_filename, delimiter=',', dtype='object')
        X = data[:, :-1]
        y = data[:, -1]
        rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits)

        for fold_idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
            for clf_idx, clf in enumerate(clfs.values()):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                scores[dataset_idx, fold_idx, clf_idx] = balanced_accuracy_score(y_test, y_pred) 

    means = np.mean(scores, axis=1)
    alpha = 0.05

    s, p = wilcoxon(means[:, 0], means[:, 1])

    print("GNB results: %f (%f)" % (round(np.mean(means[:, 0], axis=0), 3), round(np.std(means[:, 0], axis=0), 3)))
    print("KNN results: %f (%f)" % (round(np.mean(means[:, 1], axis=0), 3), round(np.std(means[:, 1], axis=0), 3)))
    print("statistic: %f, p-value: %f" % (round(s, 3), round(p, 3)))
    print("Statistically significant difference: %s" % bool(p < alpha))

    print("-" * 50)

if __name__ == "__main__":
    task1()
    task2()
    task3()
