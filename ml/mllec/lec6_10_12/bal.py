import numpy as np

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone, BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode

class BorderAL(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator: BaseEstimator = GaussianNB(), budget: float = 0.01):
        self.base_estimator = base_estimator
        self.budget = budget

    def fit(self, X_seed, y_seed, X_unlabeled, y_unlabeled):
        self.classes = unique_labels(y_seed)

        self.X_seed = X_seed
        self.y_seed = y_seed
        self.X_unlabeled = X_unlabeled
        self.y_unlabeled = y_unlabeled

        self.clf_ = clone(self.base_estimator).fit(X_seed, y_seed)
        self.n_queries = int(self.budget * X_unlabeled.shape[0])

        probas = self.clf_.predict_proba(X_unlabeled)
        self.distances = np.argsort(np.abs(probas[:, 0] - 0.5))[:self.n_queries]
        # self.distances = np.flip(np.argsort(np.abs(probas[:, 0] - 0.5))[:self.n_queries])
        
        self.clf_.partial_fit(X_unlabeled[self.distances], y_unlabeled[self.distances], classes=self.classes)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self.clf_.predict(X)
