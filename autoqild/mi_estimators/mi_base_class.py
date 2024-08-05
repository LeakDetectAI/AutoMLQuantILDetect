import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


class MIEstimatorBase(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, n_features, random_state):
        self.logger = logging.getLogger(name=MIEstimatorBase.__name__)
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = check_random_state(random_state)

    def fit(self, **kwd):
        pass

    def predict(self, X, verbose=0):
        pass

    def score(self, X, y, sample_weight=None, verbose=0):
        pass

    def predict_proba(self, X, verbose=0):
        pass

    def decision_function(self, X, verbose=0):
        pass

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
