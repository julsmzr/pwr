# Showcasing simple classifier on synthetic data

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def fit_to_data(clf):
    clf.fit(X, y)
    return clf

def predict(clf):

    # regular prediction
    preds = clf.predict(X)
    return accuracy_score(y, preds)

# visualize the line of the prediction
# preds = clf.predict_proba(X) # class assignemnt confidence (100,2)

# we find line by finding distance to 0.5 (where the distance is the same for each of the probs)
# dist = abs(preds[:, 0] - 0.5) # col1 will have same vals

# to see the line we equally sample the whole space and assign the distance to the line as a color. (heatmap)
# e.g. np.linspace(0, 1, 10) # uniform sampling of the space inlcusing borders

def visualize_pred(clf):

    n = 100
    nx = np.linspace(X[:, 0].min(), X[:, 0].max(), n)
    ny = np.linspace(X[:, 1].min(), X[:, 1].max(), n)

    # visualize using np.meshgrid
    xx, yy = np.meshgrid(nx, ny)
    map = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1) # one column each -> new dataset with the uniform sampled featues

    # now predict on the new data!
    clf.fit(X, y)
    preds = clf.predict_proba(map)
    distance = preds[:, 0] - 0.5

    # visualize data and heatmap
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.grid(ls=":", c=(.7, .7, .7))

    ax.scatter(map[:, 0], map[:, 1], c=distance, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr_r")

    plt.tight_layout()
    plt.savefig("outputs/synthetic_heatmap.png", dpi=300)

if __name__ == "__main__":

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=100, n_features=2, 
        n_informative=2, n_redundant=0, 
        n_repeated=0, random_state=None
        )

    # clf = fit_to_data(LogisticRegression())
    # clf = fit_to_data(GaussianNB())
    
    clf = fit_to_data(MLPClassifier(max_iter=1000))
    print("Prediction Accuracy:", predict(clf))

    visualize_pred(clf)
