from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

X, y = make_regression(n_samples=10, n_features=3, n_informative=3, n_targets=1, random_state=None)
X = X[:, 0]
y = y/10

aa = list(range(0, 20, 1))
bb = [-4]

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
XX = np.linspace()

for a, b in itertools.product(aa, bb):

    SR = np.sum((a*X+b - y) ** 2)
    SR_line.append(SR)
    plt.

    # TODO complete code after upload