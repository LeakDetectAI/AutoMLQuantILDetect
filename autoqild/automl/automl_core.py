from abc import abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin


class AutomlClassifier(BaseEstimator, ClassifierMixin):
    """
        Abstract base class for AutoML classifiers.

        This class serves as a base for implementing various AutoML classifiers.
        It inherits from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`,
        providing standard interfaces for scikit-learn estimators and classifiers.

        The class is abstract, meaning it cannot be instantiated directly.
        Subclasses must implement the `fit` method to provide functionality for training the classifier.

        Attributes
        ----------
        None

        Examples
        --------
        To create a custom AutoML classifier, subclass `AutomlClassifier` and implement the `fit` method:

        >>> class CustomClassifier(AutomlClassifier):
        >>>     def fit(self, X, y, **kwd):
        >>>         # Implement your fitting logic here
        >>>         pass
    """
    @abstractmethod
    def fit(self, X, y, **kwd):
        """
            Fit the AutoML classifier on the provided dataset.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data.

            y : array-like of shape (n_samples,)
                Target values.

            **kwd : keyword arguments
                Additional parameters for the fit method.

            Returns
            -------
            self : object
                Returns the instance itself.

            Notes
            -----
            This method must be implemented by subclasses. It should contain the logic
            for training the classifier on the dataset provided in `X` and `y`.
        """
        pass

    @abstractmethod
    def score(self, X, y, sample_weight=None, verbose=0):
        """
            Return the score based on the metric on the given test data and labels.

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
        """
        pass

    @abstractmethod
    def predict(self, X, verbose=0):
        """
            Predict class labels for samples in X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.

            Returns
            -------
            y_pred : array-like of shape (n_samples,)
                Predicted class labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X, verbose=0):
        """
           Predict class probabilities for samples in X.

           Parameters
           ----------
           X : array-like of shape (n_samples, n_features)
               Input samples.

           Returns
           -------
           y_proba : array-like of shape (n_samples, n_classes)
               Predicted class probabilities.
        """
        pass

    @abstractmethod
    def decision_function(self, X, verbose=0):
        """
           Predict confidence scores for samples, sometimes conincciding with the probability scores in X.
           The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane.

           Parameters
           ----------
           X : array-like of shape (n_samples, n_features)
               Input samples.

           Returns
           -------
           y_proba : array-like of shape (n_samples, n_classes)
               Predicted class probabilities.
        """
        pass

    def get_params(self, deep=True):
        """
            Get parameters for this estimator.

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
        """
            Set the parameters of this estimator.

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
