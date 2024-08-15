"""Classes implementing classifiers which predicts a constant function which
predict label only using the outputs of the dataset, which are used as
baselines for detecting information leakage."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.utils import check_random_state

__all__ = ["RandomClassifier", "MajorityVoting", "PriorClassifier"]


class RandomClassifier(DummyClassifier):
    """A classifier that predicts classes randomly according to a uniform
    distribution.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments to pass to DummyClassifier.
    """
    def __init__(self, **kwargs):
        super(RandomClassifier, self).__init__(strategy="uniform", **kwargs)


class MajorityVoting(DummyClassifier):
    """A classifier that always predicts the most frequent class.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments to pass to DummyClassifier.
    """
    def __init__(self, **kwargs):
        super(MajorityVoting, self).__init__(strategy="most_frequent", **kwargs)


class PriorClassifier(DummyClassifier):
    """PriorClassifier is a simple classifier that predicts class labels based
    on the prior distribution of the classes in the training data. This
    classifier is essentially a wrapper around the `DummyClassifier` from
    scikit-learn with a strategy set to `prior`.

    Parameters
    ----------
    random_state : int or None, optional, default=None
        Random state for reproducibility.

    **kwargs : dict, optional
        Additional keyword arguments to pass to `DummyClassifier`.

    Attributes
    ----------
    class_probabilities : array-like of shape (n_classes,)
        The prior probabilities of each class, calculated from the training data.

    classes_ : array-like of shape (n_classes,)
        The unique classes found in the training data.

    n_classes : int
        The number of unique classes in the training data.

    random_state : RandomState
        Random state instance for reproducibility.
    """
    def __init__(self, random_state=None, **kwargs):
        super(PriorClassifier, self).__init__(strategy="prior", **kwargs)
        self.class_probabilities = [0.5, 0.5]
        self.classes_ = [0, 1]
        self.n_classes = 2
        self.random_state = check_random_state(random_state)

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : PriorClassifier
            Fitted estimator.
        """
        super(PriorClassifier, self).fit(X, y, sample_weight=None)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.class_probabilities = np.zeros(self.n_classes) + 1 / self.n_classes
        for i in self.classes_:
            self.class_probabilities[i] = len(y[y == i]) / len(y)

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        n = X.shape[0]
        y_pred = self.random_state.choice(self.classes_, p=self.class_probabilities, size=n)
        return y_pred