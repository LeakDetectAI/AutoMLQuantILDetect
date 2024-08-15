"""Base class for all Mutual Information estimators."""
import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


class MIEstimatorBase(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, n_features, random_state):
        """Base class for all Mutual Information estimators.

        Parameters
        ----------
        n_classes : int
            Number of classes in the classification data samples.

        n_features : int
            Number of features or dimensionality of the inputs of the classification data samples.

        random_state : int or object, optional, default=42
            Random state for reproducibility.

        Attributes
        ----------
        logger : logging.Logger
            Logger instance for logging information.
        """
        self.logger = logging.getLogger(name=MIEstimatorBase.__name__)
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = check_random_state(random_state)

    def fit(self, **kwd):
        """Fit the mutual information estimation model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        **kwd : dict, optional
            Additional keyword arguments passed to the `fit` method of the classifier.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        raise NotImplementedError("The 'fit' method must be implemented by the subclass.")

    def score(self, X, y, sample_weight=None, verbose=0):
        """Return the score based on the metric on the given test data and
        labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The 'score' method must be implemented by the subclass.")

    def predict(self, X, verbose=0):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The 'predict' method must be implemented by the subclass.")

    def predict_proba(self, X, verbose=0):
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Input samples.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
           Predicted class probabilities.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The 'predict_proba' method must be implemented by the subclass.")

    def decision_function(self, X, verbose=0):
        """Predict confidence scores for samples, sometimes conincciding with
        the probability scores in X. The confidence score for a sample is
        proportional to the signed distance of that sample to the hyperplane.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The 'decision_function' method must be implemented by the subclass.")

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def estimate_mi(self, X, y, **kwargs):
        """Estimate Mutual Information using the specified method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        **kwargs : dict, optional
            Additional keyword arguments passed to the estimation methods.

        Returns
        -------
        mutual_information : float
            A mean of estimated MI values from cross-validation splits.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The 'estimate_mi' method must be implemented by the subclass.")
