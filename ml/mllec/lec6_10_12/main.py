import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB

from al import RandomAL
from bal import BorderAL
from cal import CombinedAL


def load_dataset():
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=0, 
                               n_repeated=0, n_informative=10, random_state=1410, 
                               class_sep=0.4)
    return X, y


def perform_rskf(clf_template, X, y, is_active: bool, is_full: bool):
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
    scores = []

    for train_idx, test_idx in rskf.split(X, y):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_test_fold, y_test_fold = X[test_idx], y[test_idx]
        
        clf = clone(clf_template)

        X_seed, X_unlabeled, y_seed, y_unlabeled = train_test_split(
            X_train_fold, y_train_fold, 
            test_size=0.95,
            random_state=42, 
            stratify=y_train_fold
        )

        if is_active:
            clf.fit(X_seed, y_seed, X_unlabeled, y_unlabeled)
        elif is_full:
            clf.fit(X_train_fold, y_train_fold)
        else:
            clf.fit(X_seed, y_seed)

        preds = clf.predict(X_test_fold)
        scores.append(accuracy_score(y_test_fold, preds))

    return np.mean(scores)


if __name__ == "__main__":
    X, y = load_dataset()
    
    budgets = np.arange(0.01, 0.96, 0.05)
    
    clf_names = [
        "GaussianNB", 
        "RandomAL", 
        "BorderAL", 
        "CombinedAL", 
        "GaussianNB (full)"
    ]
    
    # BUDGETS x CLFs
    scores = np.zeros((len(budgets), len(clf_names)))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for budget_idx, budget in enumerate(budgets):        
        clfs = [
            GaussianNB(),
            RandomAL(base_estimator=GaussianNB(), budget=budget, random_state=42),
            BorderAL(base_estimator=GaussianNB(), budget=budget),
            CombinedAL(base_estimator=GaussianNB(), budget=budget, budget_split_ratio=0.5, random_state=42),
            GaussianNB() 
        ]
        
        for i, clf in enumerate(clfs):
            name = clf_names[i]
            is_full = "full" in name
            is_active = "AL" in name and not is_full

            scores[budget_idx, i] = perform_rskf(clf, X, y, is_active=is_active, is_full=is_full)

    for i, name in enumerate(clf_names):
        ax.plot(budgets, scores[:, i], label=name)

    ax.set_xlabel("Budget")
    ax.set_ylabel("Accuracy on X_test")
    ax.grid(ls=":", c=(0.7, 0.7, 0.7))
    ax.set_ylim(0.5, 1)
    ax.legend()

    fig.tight_layout()
    plt.savefig("outputs/plot.png", dpi=300)
