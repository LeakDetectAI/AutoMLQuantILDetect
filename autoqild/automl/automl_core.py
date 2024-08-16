"""Abstract base class for AutoML classifiers."""

from abc import abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin


class AutomlClassifier(BaseEstimator, ClassifierMixin):
    """Abstract base class for AutoML classifiers.

    This class serves as a base for implementing various AutoML classifiers.
    It inherits from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`,
    providing standard interfaces for scikit-learn estimators and classifiers.

    The class is abstract, meaning it cannot be instantiated directly.
    Subclasses must implement the 'fit' method to provide functionality for training the classifier.

    Examples
    --------
    To create a custom AutoML classifier, you can subclass `AutomlClassifier` and implement the required methods,
    such as 'fit', 'fit', and 'fit'. Below is an example of how you might implement a simple custom classifier.

    First, create a subclass of `AutomlClassifier`:

    >>> from sklearn.pipeline import Pipeline
    >>> class CustomClassifier(AutomlClassifier):
    >>>     def fit(self, X, y, **kwd):
    >>>         # Implement your fitting logic here
    >>>         # For instance, you might train a model using the training data X and labels y
    >>>         self.model_ = Pipeline([("scaler", StandardScaler()), ("custom_classifier", CustomClassifier())])
    >>>         return self
    >>>
    >>>     def predict(self, X, verbose=0):
    >>>         # Implement your prediction logic here
    >>>         # For example, use the trained model to make predictions on new data X
    >>>         return self.model_.predict(X)
    >>>
    >>>     def score(self, X, y, sample_weight=None, verbose=0):
    >>>         # Implement your scoring logic here
    >>>         # For instance, calculate the accuracy of predictions compared to true labels y
    >>>         predictions = self.predict(X)
    >>>         return accuracy_score(y, predictions)
    >>>
    >>>     def predict_proba(self, X, verbose=0):
    >>>         # Implement your probability prediction logic here
    >>>         # For example, return the predicted probabilities for each class
    >>>         return self.model_.predict_proba(X)
    >>>
    >>>     def decision_function(self, X, verbose=0):
    >>>         # Implement your decision function logic here
    >>>         # For instance, return the decision function values (e.g., distances to decision boundary)
    >>>         return self.model_.decision_function(X)

    After defining your custom classifier, you can use it like any other scikit-learn estimator:

    >>> clf = CustomClassifier()
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> accuracy = clf.score(X_test, y_test)
    """

    @abstractmethod
    def fit(self, X, y, **kwd):
        """Fit the AutoML classifier on the provided dataset.

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

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.

        Notes
        -----
        This method must be implemented by subclasses. It should contain the logic
        for training the classifier on the dataset provided in `X` and `y`.
        """

        raise NotImplementedError(
            "The 'fit' method must be implemented by the subclass."
        )

    @abstractmethod
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

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` with respect to `y`.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError(
            "The 'fit' method must be implemented by the subclass."
        )

    @abstractmethod
    def predict(self, X, verbose=0):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError(
            "The 'predict' method must be implemented by the subclass."
        )

    @abstractmethod
    def predict_proba(self, X, verbose=0):
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError(
            "The 'predict_proba' method must be implemented by the subclass."
        )

    @abstractmethod
    def decision_function(self, X, verbose=0):
        """Predict confidence scores for samples, sometimes coinciding with the
        probability scores in X. The confidence score for a sample is
        proportional to the signed distance of that sample to the hyperplane.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Predicted confidence scores.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError(
            "The 'decision_function' method must be implemented by the subclass."
        )

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
        (such as `sklearn.pipeline.Pipeline`). The latter have
        parameters of the form `<component>__<parameter>` so that it`s
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
