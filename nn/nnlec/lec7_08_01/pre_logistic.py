import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    random_state=1410,
    n_clusters_per_class=1
)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def log_odds(probability):
    return np.log(probability / (1 - probability))

def probability(log_odds):
    return (np.exp(log_odds)) / ((1 + log_odds))

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.ravel()

# probability plot
ax[0].scatter(X, y, c=y, cmap='bwr')
ax[0].grid(ls=":", c=(.7, .7, .7))


# left:
# Sigmoid Function
# Sigmoid Function | Maximum Likelihood

# on the right: log odds function each




fig.savefig("pre_logistic.png", dpi=300)


