# General introduction to some numpy basics, data visualization and dimensionality reduction

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def print_dataset(X, y):
    print(X) # data (samples, features)
    print(X.shape) # (num_rows, num_cols)
    print(y) # labels
    print(y.shape)

def plot_two_features(X, y, cols):
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) #num_rows, num_cols, figsize

    # plot featurespace (show first two of four features)
    ax.scatter(X[:, 0], X[:, 1], c=y) # plot first col and second col, color is label y (type of flower)

    # label the plot axes 
    ax.set_ylabel(cols[0], fontsize=16)
    ax.set_xlabel(cols[1], fontsize=16)

    plt.tight_layout() # cut off dataless areas
    plt.savefig("outputs/tabular_two_features.png")


# issue: cannot plot four dimensional full feature space
# first way to solve this: plot features pairs (each one to each one.)

def plot_feature_pairs(X, y, cols):
    n_features = X.shape[1]
    fig, ax = plt.subplots(n_features, n_features, figsize=(16, 16)) 

    for i in range(n_features):
        for j in range(n_features):
            ax[i, j].scatter(X[:, i], X[:, j], c=y) # type: ignore
            ax[i, j].set_ylabel(cols[i]) # type: ignore
            ax[i, j].set_xlabel(cols[j]) # type: ignore

    plt.tight_layout()
    plt.savefig("outputs/tabular_feature_pairs.png")

# for very high dimensionality this is not feasible (curse of dimensionality)
# -> we could use PCA: squeeze feature space into lower dimensionality
# result: low dims but lose a bit of information, also dims lose meaning

def plot_pca(X, y):
    pca = PCA(n_components=2) # 2 is target dimensions
    X = pca.fit_transform(X) # type: ignore

    print(pca.explained_variance_ratio_) 
    # [0.92461872 0.05306648] 
    # tells how much information is encased in these new dimensions
    # -> we have lost about 2% of information

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.grid(ls=":", c=(.7, .7, .7))

    plt.tight_layout()
    plt.savefig("outputs/tabular_pca.png", dpi=300)

if __name__ == "__main__":

    # Toy dataset from scikit-learn showcasing flowers
    X, y = load_iris(return_X_y=True) # X features, y labels
    cols=("Sepal Length", "Sepal Width", "Petal Length", "Petal Width")

    os.makedirs("outputs", exist_ok=True)

    # print_dataset(X, y)
    # plot_two_features(X, y, cols)
    # plot_feature_pairs(X, y, cols)
    plot_pca(X, y)
