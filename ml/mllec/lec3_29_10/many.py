# train some classifiers on many datasets

import numpy as np
import os
from tqdm import tqdm
from tabulate import tabulate

from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score as bac


clfs = {
    "GNB": GaussianNB(),
    # "LR": LogisticRegression(),
    "CART": DecisionTreeClassifier(random_state=1410),
    "kNN": KNeighborsClassifier(),
}

datasets = {i: name for i, name in enumerate(os.listdir("datasets/classification"))}

# DATASETS x CLFS x FOLDS
scores = np.zeros((len(datasets), len(clfs), 5 * 2))
for dataset_id, dataset in tqdm(datasets.items()):
    try:
        data = np.genfromtxt(os.path.join("datasets/classification", dataset, "%s.csv" % dataset), 
                            delimiter=",", dtype=float)
        X, y = data[1:, :-1], data[1:, -1]
            
        if np.isnan(X).any() or np.isnan(y).any():
            # print("skipped", dataset, "contains nan values")
            continue

        y = y.astype(int)

        min_class_count = np.min(np.bincount(y))
        if min_class_count < 5:
            # print(f"skipped {dataset} - class with only {min_class_count} samples")
            continue
        
    except (ValueError, TypeError):
        # print("skipped", dataset, "contains non-numeric data")
        continue

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        if len(np.unique(y_train)) == 1:
            # print(f"Warning: Only one class in training data: {np.unique(y_train)}")
            continue

        for clf_id, clf_name in enumerate(clfs.keys()):
            clf = clone(clfs[clf_name])
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            if len(np.unique(preds)) == 1:
                # print(f"All predictions are: {preds[0]}")
                continue

            score = bac(y_test, preds)
            scores[dataset_id, clf_id, i] = score


trained_indices = np.where(np.any(scores != 0, axis=(1, 2)))[0]
print("Trained on %s datasets" % len(trained_indices))
for i in trained_indices[:3]: 
    print(f"\nScores for dataset {datasets[i]}\n", tabulate(scores[i], showindex=list(clfs.keys())))

# Save which datasets were actually used
trained_datasets = {i: datasets[i] for i in trained_indices}
np.save("data/trained_datasets.npy", np.array(trained_datasets))
np.save("data/scores_many.npy", scores)
