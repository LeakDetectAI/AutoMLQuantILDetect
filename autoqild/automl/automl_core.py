from abc import abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin


class AutomlClassifier(BaseEstimator, ClassifierMixin):
    @abstractmethod
    def fit(self, X, y, kwd):
        pass

    @abstractmethod
    def predict(self, X, verbose):
        pass

    @abstractmethod
    def score(self, X, y, sample_weight, verbose):
        pass

    @abstractmethod
    def predict_proba(self, X, verbose):
        pass

    @abstractmethod
    def decision_function(self, X, verbose):
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
