import numpy as np

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone, BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted # type: ignore
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode

class RandomSubspace(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator: BaseEstimator = GaussianNB(), n_estimators = 10, n_features=2, voting="hard", random_state = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.voting = voting
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.subspaces = self.random.choice(X.shape[1], size=(self.n_estimators, self.n_features), replace=True)
        self.ensemble = [
            clone(self.base_estimator).fit(self.X_[:, self.subspaces[i]], y)
            for i in range(self.n_estimators)
            ]
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        self.preds_arr = np.array([
            self.ensemble[i].predict(X[:, self.subspaces[i]])
            for i in range(self.n_estimators)
        ])

        if self.voting == "hard": # majority voting
            self.preds = mode(self.preds_arr, axis=0)[0]
        elif self.voting == "soft": # 
            self.probas_arr = np.array([
            self.ensemble[i].predict_proba(X[:, self.subspaces[i]])
            for i in range(self.n_estimators)
        ])
            self.mean_probas = np.mean(self.probas_arr, axis=0)
            self.preds = np.argmax(self.mean_probas, axis=1)

        return self.preds

# TODO clean