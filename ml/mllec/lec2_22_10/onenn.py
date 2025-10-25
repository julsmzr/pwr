# How to build simple 1-NN classifier

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.utils.validation import validate_data, check_is_fitted # type: ignore
from sklearn.metrics import DistanceMetric


class OneNN(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.distance = DistanceMetric.get_metric('euclidean')

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X):
        # check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        # distance_matrix = np.sort(distance_matrix, axis=1) 
        # first val is min dist. for expansion we could just select first second third for e.g. 3-nn

        # indices of sorted elements, only take first col (1-nn)
        distance_matrix = np.argsort(self.distance.pairwise(X, self.X_), axis=1)[:, 0] 
        preds = self.y_[distance_matrix] # use the col of indices 

        return preds
