# train some classifiers on a single dataset

import numpy as np
from tabulate import tabulate

from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score as bac


data = np.genfromtxt("datasets/classification/australian/australian.csv", delimiter=",")
X, y = data[1:, :-1], data[1:, -1].astype(int)

clfs = {
    "GNB": GaussianNB(),
    # "LR": LogisticRegression(),
    "CART": DecisionTreeClassifier(random_state=1410),
    "kNN": KNeighborsClassifier(),
}

# CLFS x FOLDS
scores = np.zeros((len(clfs), 5 * 2))

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    for clf_id, clf_name in enumerate(clfs.keys()):
        clf = clone(clfs[clf_name])
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        score = bac(y_test, preds)
        scores[clf_id, i] = score

print("\nScores\n", tabulate(scores, showindex=list(clfs.keys())), "\n")
np.save("data/scores_single.npy", scores)
