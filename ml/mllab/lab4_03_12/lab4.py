import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import rand_score, completeness_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

np.random.seed(33)

def task1():
    print("\nTask 1")
    print("-" * 50)

    X, y = make_blobs(n_samples=500, centers=3, cluster_std=3.0, random_state=33)
    y_pred = KMeans(n_clusters=3).fit_predict(X)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].scatter(X[:, 0], X[:, 1], c='black')
    ax[1].scatter(X[:, 0], X[:, 1], c=y)
    ax[2].scatter(X[:, 0], X[:, 1], c=y_pred)

    for i in range(3):
        ax[i].grid(ls=":", c=(.7, .7, .7))

    fig.tight_layout()
    fig.savefig("outputs/task1.png", dpi=300)

    print("Figure saved to outputs/task1.png")
    print("-" * 50)

def task2():
    print("\nTask 2")
    print("-" * 50)

    class CustomKMeans(BaseEstimator, ClusterMixin):

        def __init__(self, k: int, iters: int) -> None:
            self.k = k
            self.iters = iters

        def fit_predict(self, X: np.ndarray) -> np.ndarray:
            centers = np.random.uniform(-5, 5, size=6).reshape(3,2)

            for i in range(self.iters):
                print("Iteration %s:" % i)

                d = cdist(X, centers)
                a = np.argmin(d, axis=1)

                for i in range(3):
                    centers[i] = np.mean(X[a == i], axis=0)

                print(centers, "\n")
            
            return np.argmin(cdist(X, centers), axis=1)

    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=3.0, random_state=33)

    y_pred_scikit = KMeans(n_clusters=3).fit_predict(X)
    y_pred_custom = CustomKMeans(k=3, iters=5).fit_predict(X)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].scatter(X[:, 0], X[:, 1], c=y_pred_scikit)
    ax[1].scatter(X[:, 0], X[:, 1], c=y_pred_custom)

    for i in range(2):
        ax[i].grid(ls=":", c=(.7, .7, .7))

    fig.tight_layout()
    fig.savefig("outputs/task2.png", dpi=300)

    print("Figure saved to outputs/task2.png")
    print("-" * 50)

def task3():
    print("\nTask 3")
    print("-" * 50)

    random_center = np.random.randint(3, 10)
    X, y = make_blobs(n_samples=1500, n_features=10, cluster_std=1.0, centers=random_center, random_state=33)

    low_k = 2
    high_k = 11

    cardinalities = np.linspace(low_k, high_k, high_k - low_k + 1, dtype=int)
    scores = np.zeros((2, 2, len(cardinalities)))

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))

    for k_idx, k in enumerate(cardinalities):
        y_pred = KMeans(n_clusters=k).fit_predict(X)

        scores[0, 0, k_idx] = (rand_score(y, y_pred))
        scores[0, 1, k_idx] = (completeness_score(y, y_pred))
        scores[1, 0, k_idx] = (davies_bouldin_score(X, y_pred))
        scores[1, 1, k_idx] = (calinski_harabasz_score(X, y_pred))

    for i in range(2):
        for j in range(2):
            ax[i, j].plot(cardinalities, scores[i, j], color='black', marker='o')
            ax[i, j].scatter(np.argmax(scores[i, j]) + 2, np.max(scores[i, j]), marker='x', color='red', s=100, zorder=10)
            ax[i, j].grid(ls=":", c=(.7, .7, .7))

    fig.tight_layout()
    fig.savefig("outputs/task3.png", dpi=300)

    print("Random number of centers:", random_center)
    print("\nFigure saved to outputs/task3.png")
    print("-" * 50)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    task1()
    task2()
    task3()
