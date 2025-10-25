# analyse the results of three classifiers trained on a single dataset

import numpy as np
from tabulate import tabulate

from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
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

# CLFS x FOLDS
scores = np.load("data/scores_single.npy")

alpha = 0.05 
t_statistic = np.zeros((len(clfs), len(clfs))) 
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = cv52cvt(scores[i], scores[j])
        # t_statistic[i, j], p_value[i, j] = wilcoxon(scores[i], scores[j]) # type: ignore
        # t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

print("\nT Statistic")
print('-' * 30)
print(tabulate(t_statistic, tablefmt="simple", floatfmt=".3f", showindex=list(clfs.keys()), headers=list(clfs.keys())))
print('-' * 30)

# show advantage comparison from,to
mean_scores = np.mean(scores, axis=1)
advantage = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        if i != j:
            advantage[i, j] = 1 if mean_scores[i] > mean_scores[j] else 0

print("\nAdvantage")
print('-' * 30)
print(tabulate(advantage, tablefmt="simple", floatfmt=".0f", showindex=list(clfs.keys()), headers=list(clfs.keys())))
print('-' * 30)

# check if difference is statistically significant
significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alpha] = 1

print("\nSignificance")
print('-' * 30)
print(tabulate(significance, tablefmt="simple", floatfmt=".0f", showindex=list(clfs.keys()), headers=list(clfs.keys())))
print('-' * 30)

# then: advantage & significance (which one is significantly better compared to which one)
stat_better = significance * advantage
# print(np.mean(scores, axis=1)) # e.g. scores: 0.78, 0.81, 0.81, 0.65 -> GNB significantly better than KNN 0.78 >> 0.65 etc.

print("\nStatistically Better")
print('-' * 30)
print(tabulate(stat_better, tablefmt="simple", floatfmt=".0f", showindex=list(clfs.keys()), headers=list(clfs.keys())))
print('-' * 30)
print("\n")

print("Mean scores:")
for name, score in zip(clfs.keys(), np.mean(scores, axis=1)):
    print(f"{name}: {score:.4f}")
