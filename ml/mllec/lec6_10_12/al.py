import numpy as np

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone, BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode

class RandomAL(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator: BaseEstimator = GaussianNB(), budget: float = 0.01, random_state = None):
        self.base_estimator = base_estimator
        self.budget = budget

        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

    def fit(self, X_seed, y_seed, X_unlabeled, y_unlabeled):
        self.classes = unique_labels(y_seed)

        self.X_seed = X_seed
        self.y_seed = y_seed
        self.X_unlabeled = X_unlabeled
        self.y_unlabeled = y_unlabeled

        self.clf_ = clone(self.base_estimator).fit(X_seed, y_seed)
        self.n_queries = int(self.budget * X_unlabeled.shape[0])
        
        self.indices = self.random.choice(
            X_unlabeled.shape[0], 
            size=self.n_queries,
            replace=False
            )
        
        self.clf_.partial_fit(X_unlabeled[self.indices], y_unlabeled[self.indices], classes=self.classes)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self.clf_.predict(X)
