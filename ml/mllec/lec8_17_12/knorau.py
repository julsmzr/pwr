import numpy as np

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone, BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, DistanceMetric
from scipy.stats import mode

class KnoraU(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator: BaseEstimator = GaussianNB(), n_estimators = 10, voting="hard", k=7, random_state = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting
        self.k = k

        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

        self.dist = DistanceMetric().get_metric("euclidean")

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # sampling with replacement
        self.estimators_ = []
        indxs = self.random.choice(self.X_.shape[0], (self.n_estimators, X.shape[0]), replace=True)
        for i in range(self.n_estimators):
            self.estimators_.append(clone(self.base_estimator).fit(self.X_[indxs[i]], self.y_[indxs[i]]))
        
        return self
    
    def competence(self, X, X_DSEL, y_DSEL):
        """8======D-->"""
        ditstance_matrix = self.dist.pairwise(X, X_DSEL)
        local_comp_indxs = np.argsort(ditstance_matrix, axis=1)[:, :self.k]
        X_local = X_DSEL[local_comp_indxs]
        y_local = y_DSEL[local_comp_indxs]

        estimator_weights = []
        for estimator in self.estimators_:
            local_preds = np.array([estimator.predict(X_local[i]) for i in range(X_local.shape[0])])
            estimator_weights.append(np.sum((local_preds[0] == y_local).astype(int), axis=1))

        return np.array(estimator_weights)

        
    def predict(self, X, X_DSEL, y_DSEL):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        weights = self.competence(X, X_DSEL, y_DSEL)
        test_preds = np.array([estimator.predict(X) for estimator in self.estimators_])
        self.preds = []

        for i, test_sample_preds in enumerate(test_preds.T):
            count = np.bincount(test_sample_preds, weights=weights.T[i])
            # count = np.bincount(test_sample_preds, weights=None)
            self.preds.append(np.argmax(count))

        return self.preds
