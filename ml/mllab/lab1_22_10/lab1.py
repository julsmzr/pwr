import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def load_data(print_shape = False):
    data = np.loadtxt("data/iris.csv", delimiter=',', dtype='object')
    if print_shape:
        print(data.shape)

    column_names = data[0][:4] # remove species (last col)
    data = data[1:]

    X = data[:, :4]
    y = data[:, 4]

    y[y == 'setosa'] = 0
    y[y == 'versicolor'] = 1
    y[y == 'virginica'] = 2

    X = X.astype(float)
    y = y.astype(int)

    color_map = {0: "r", 1: "g", 2: "b"}
    colors = []
    for label in y:
        colors.append(color_map[label])

    return X, y, column_names, colors

def task1():
    X, y, column_names, colors = load_data(print_shape=True)

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.scatter(X[:, 0], X[:, 1], c=colors)

    ax.set_xlabel(column_names[0])
    ax.set_ylabel(column_names[1])

    fig.tight_layout()
    fig.savefig("outputs/task1.png", dpi=300)


def task2():
    X, y, column_names, colors = load_data()

    fig, ax = plt.subplots(4, 4, figsize=(18, 18))
    for i in range(4):
        for j in range(4):
            ax[i, j].scatter(X[:, i], X[:, j], c=colors)
            ax[i, j].set_xlabel(column_names[i])
            ax[i, j].set_ylabel(column_names[j])

    fig.tight_layout()
    fig.savefig("outputs/task2.png", dpi=300)


def task3():
    X, y, column_names, colors = load_data()
    X = X[:, 2:]
    column_names = column_names[2:]

    new_sample = np.asarray([[3.1, 1.2]])

    centroids = []
    for i in range(3):
        centroids.append(np.mean(X[y==i], axis=0))
    centroids = np.asarray(centroids)

    distances = cdist(new_sample, centroids)
    print("Distances:", distances)
    
    closest_match = np.argmin(distances)
    class_names = {0: "setosa", 1: "versicolor", 2: "virginicia"}
    print("Prediction:", class_names[int(closest_match)])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.15)
    ax.scatter(centroids[:, 0], centroids[:, 1], c=["r", "g", "b"])
    ax.scatter(new_sample[:, 0], new_sample[:, 1], marker="x", c="black")

    ax.set_xlabel(column_names[0])
    ax.set_ylabel(column_names[1])
    ax.grid(ls=":", c=(.7, .7, .7))

    fig.tight_layout()
    fig.savefig("outputs/task3.png", dpi=300)


if __name__ == "__main__":
    # task1() 
    # task2()
    task3()
