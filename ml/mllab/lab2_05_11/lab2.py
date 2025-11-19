import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def task1():

    print("\nTask 1")
    print("-" * 50)

    X, y = make_classification(n_samples=700, n_features=10, flip_y=0.08)
    print("Data shapes:", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Training set shapes:", X_train.shape, y_train.shape)
    print("Testing set shapes:", X_test.shape, y_test.shape)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy score:", round(acc, 3))

    print("-" * 50)

def task2():
    
    print("\nTask 2")
    print("-" * 50)

    X, y = make_classification(n_samples=700, n_features=10, flip_y=0.08)

    n_splits = 2
    n_repeats = 5

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    # FOLDS
    scores = np.full(shape=(n_splits * n_repeats), fill_value=np.nan)

    for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        clf = clone(GaussianNB())
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores[fold_idx] = score    

    print("All results:")
    print(np.round(scores, 3))

    mn = np.mean(scores)
    std = np.std(scores)

    print("Mean value:", round(mn, 3))
    print("Standard deviation", round(std, 3))

    print("-" * 50)

def task3(plot: bool = False):
    
    print("\nTask 3")
    print("-" * 50)

    datasets = [
        make_classification(n_samples=500, n_features=5),
        make_classification(n_samples=1000, n_features=10),
        make_classification(n_samples=1500, n_features=25, class_sep=0.5),
    ]
    n_datasets = len(datasets)

    clfs = [
        GaussianNB(),
        KNeighborsClassifier()
    ]
    n_clfs = len(clfs)

    n_splits = 2
    n_repeats = 5

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    # DATASETS x FOLDS x CLFS
    scores = np.full((n_datasets, n_splits * n_repeats, n_clfs), np.nan)

    for dataset_idx, (X, y) in enumerate(datasets):
        for fold_idx, (train_index, test_index) in enumerate(rkf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            for clf_idx, clf in enumerate(clfs):
                clf = clone(clf)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores[dataset_idx, fold_idx, clf_idx] = score   

    print("Average Results:")
    print(np.mean(scores, axis=1))

    print("-" * 50)
    print()

    # Bonus Task: Violin Plots
    if not plot:
        return
    
    dataset_names = ["Dataset %d" % (i+1) for i in range(n_datasets)]
    clf_names = ["GNB", "KNN"]

    fig, ax = plt.subplots(1, n_datasets, figsize=(10, 5))

    y_min = min(np.min(score) for score in scores)
    y_max = max(np.max(score) for score in scores)

    # Add some padding
    y_padding = (y_max - y_min) * 0.05
    y_min = y_min - y_padding
    y_max = y_max + y_padding

    for dataset_idx in range(n_datasets):

        if dataset_idx > 0:
            ax[dataset_idx].set_yticklabels([])

        ax[dataset_idx].violinplot(scores[dataset_idx], showmeans=True)
        ax[dataset_idx].set_title(dataset_names[dataset_idx])

        ax[dataset_idx].spines[['top', 'right']].set_visible(False)
        ax[dataset_idx].set_ylim(y_min, 1.0)

        ax[dataset_idx].grid(ls=":", c=(.7, .7, .7))

        ax[dataset_idx].set_xticks([1, 2])
        ax[dataset_idx].set_xticklabels(clf_names)

    ax[0].set_ylabel("accuracy score")

    plt.tight_layout()
    plt.savefig("outputs/task3.png", dpi=300)


if __name__ == "__main__":
    task1() 
    task2()
    task3(plot=True)
