# Showing how to work with an image dataset

import matplotlib.pyplot as plt
import numpy as np
import os
from cv2 import resize

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def load_dataset(classes=["cat", "horse"]):
    X = []
    y = []

    for class_id, cls in enumerate(classes):
        files = os.listdir(f"datasets/animals-10/{cls}")
        for file_id, filename in enumerate(files[:100]):
            img_path = f"datasets/animals-10/{cls}/{filename}"
            img = plt.imread(img_path)

            # take care of images (png) with different amount of color channels
            if img.shape[2] == 3:
                X.append(resize(img, (224, 224)))
                y.append(class_id)
            
    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])) # flatten
    return X, y

def pca_2dim(X, n_components=2):
    pca = PCA(n_components=n_components) # -> decompose into 2 dims
    X = pca.fit_transform(X)

    # print("Shape", X.shape, "Ratio", pca.explained_variance_ratio_)
    return X

def pca_targetinfo(X, targetinfo=0.7):
    pca = PCA(n_components=targetinfo) # -> decompose into n features to retain 70% if information
    X = pca.fit_transform(X)

    # print("Shape", X.shape, "Ratio", pca.explained_variance_ratio_)
    return X

def plot(X):
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr_r")
    ax.grid(ls=":", c=(.7, .7, .7))

    plt.tight_layout()
    plt.savefig("outputs/image_lowdim.png", dpi=300)

def pred(clf, X, y):
    
    clf.fit(X, y)
    pred = clf.predict(X)
    return accuracy_score(y, pred)

if __name__ == "__main__":

    # classes = os.listdir("datasets/animals-10")
    X, y = load_dataset() # X.shape: (199, 150528) -> optionally do pca to allow plotting features

    # X = pca_2dim(X)
    X = pca_targetinfo(X)
    plot(X)
    
    clf = LogisticRegression() # 0.69 with PCA, 
    # clf = GaussianNB() # 0.73 with PCA, 0.71 without
    # clf = MLPClassifier() # 1.0 with targetinfo PCA; is able to remember all images (eval on train data)

    print("Prediction Accuracy:", pred(clf, X, y))
