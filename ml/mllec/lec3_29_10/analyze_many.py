# analyse the results of three classifiers trained on many datasets

import numpy as np
import os
from tabulate import tabulate

from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, rankdata
from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# simplified of https://github.com/w4k2/weles/blob/7872d39ab1b6fe10d94253a86244150d828cb282/weles/statistics/statistics.py#L54 
def cv52cvt(a, b):
    d = a.reshape(2, 5) - b.reshape(2, 5)
    denominator = 2 * np.sum(np.var(d, axis=0, ddof=0))
    
    # Handle division by zero
    if denominator == 0:
        # If all differences are zero, F-statistic is undefined
        # Return a large p-value (no significant difference)
        return 0.0, 1.0
    
    f = np.sum(np.power(d, 2)) / denominator
    p = 1 - stats.f.cdf(f, 10, 5)
    return f, p

clfs = {
    "GNB": GaussianNB(),
    # "LR": LogisticRegression(),
    "CART": DecisionTreeClassifier(random_state=1410),
    "kNN": KNeighborsClassifier(),
}

datasets = {i: name for i, name in enumerate(os.listdir("datasets/classification"))}

# Load trained datasets mapping
trained_datasets = np.load("data/trained_datasets.npy", allow_pickle=True).item()
trained_indices = list(trained_datasets.keys())

# Filter to only trained datasets
datasets = {i: datasets[idx] for i, idx in enumerate(trained_indices)}

# Load scores and filter only trained datasets
scores = np.load("data/scores_many.npy")[trained_indices]

# rank by classifciation accruacy 
# DATASETS x CLFS
mean_scores = np.mean(scores, axis=2) # average the folds
# print(mean_scores)

# DATASETS x CLFS
std_scores = np.std(scores, axis=2) # std of the folds 
# print(std_scores)

# DATASETS x CLFS
ranked = rankdata(mean_scores, axis=1)
# print(ranked) # -> max of this shows on average which classifier was best

t = []
alpha = 0.05

# https://github.com/w4k2/weles/blob/master/weles/evaluation/PairedTests.py
for db_idx, db_name in datasets.items():
    # Row with mean scores
    t.append(["%s" % db_name[:-4]] + ["%.3f" %v for v in mean_scores[db_idx, :]])

    # Row with std; typically is not included, not needed for statistical analysis
    t.append([''] + ["%.3f" % v for v in std_scores[db_idx, :]])

    # Calculate statistic and p
    T, p = np.array(
        [[cv52cvt(scores[db_idx, i, :], scores[db_idx, j, :]) if i != j else (0.0, 1.0) for i in range(len(clfs))] for j in range(len(clfs))]
    ).swapaxes(0, 2)

    mean_adv = mean_scores[db_idx, :] > mean_scores[db_idx, :, np.newaxis]
    stat_adv = p < alpha

    _ = np.where(stat_adv * mean_adv)
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(len(clfs))]

    t.append([''] + [", ".join(["%i" % i for i in c])
                        if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else "---")
                        for c in conclusions])

print(tabulate(t, headers=[" ", "GNB", "CART", "KNN"], tablefmt="simple"))
