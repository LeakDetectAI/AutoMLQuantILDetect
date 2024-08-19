"""AutoGluon-based MI estimator that leverages automated machine learning
(AutoML) to estimate MI with optimized hyperparameters."""

from autoqild.automl import AutoGluonClassifier
from autoqild.mi_estimators.mi_estimator_classification import ClassificationMIEstimator
from autoqild.utilities._constants import LOG_LOSS_MI_ESTIMATION


class AutoMIGluonEstimator(ClassificationMIEstimator):
    """AutoMIGluonEstimator integrates the AutoGluon framework into the Mutual
    Information (MI) estimation process for classification tasks. This class
    extends the `ClassficationMIEstimator` by using AutoGluon as the base
    estimator to perform MI estimation. It allows for various configurations
    and model tuning, making it flexible for different classification
    scenarios.

    Parameters
    ----------
    n_features : int
        The number of features in the input data.

    n_classes : int
        The number of classes in the classification task.

    time_limit : int, optional, default=1800
        Time limit for training the model, in seconds.

    output_folder : str, optional
        Directory where the trained model and related files will be saved. Default is None.

    eval_metric : str, optional, default=`accuracy`
        Evaluation metric used to assess the performance of the model.

    use_hyperparameters : bool, optional, default=True
        Whether to use predefined hyperparameters for model training.

    delete_tmp_folder_after_terminate : bool, optional, default=True
        Whether to delete the temporary folder after model training is complete.

    auto_stack : bool, optional, default=True
        Whether to use automatic stacking of models in AutoGluon.

    remove_boosting_models : bool, optional, default=True
        Whether to exclude boosting models (like GBM, CAT, XGB) from the hyperparameters.

    verbosity : int, optional, default=6
        Level of verbosity for logging and output.

    random_state : int or None, optional, default=None
        Seed for random number generation to ensure reproducibility.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `AutoGluonClassifier` constructor.

    Attributes
    ----------
    base_estimator : AutoGluonClassifier
        The base AutoML estimator used for classification.

    learner_params : dict
        Parameters used to configure the base learner.

    base_learner : AutoGluonClassifier
        Instance of the AutoGluon classifier used for learning.
    """

    def __init__(
        self,
        n_features,
        n_classes,
        time_limit=1800,
        output_folder=None,
        eval_metric="accuracy",
        use_hyperparameters=True,
        delete_tmp_folder_after_terminate=True,
        auto_stack=True,
        remove_boosting_models=True,
        verbosity=6,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            n_classes=n_classes, n_features=n_features, random_state=random_state, **kwargs
        )
        self.base_estimator = AutoGluonClassifier
        self.learner_params = dict(
            n_features=n_features,
            n_classes=n_classes,
            time_limit=time_limit,
            output_folder=output_folder,
            eval_metric=eval_metric,
            use_hyperparameters=use_hyperparameters,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            auto_stack=auto_stack,
            remove_boosting_models=remove_boosting_models,
            verbosity=verbosity,
            random_state=random_state,
        )
        self.base_learner = self.base_estimator(**self.learner_params)

    def fit(self, X, y, **kwd):
        """Fit the classification model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        **kwd : dict, optional
            Additional keyword arguments passed to the `fit` method of the base learner.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super().fit(X, y, **kwd)
        return self

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
        """
        return super().predict(X, verbose=verbose)

    def score(self, X, y, sample_weight=None, verbose=0):
        """Return the accuracy score of the model on the given test data and
        labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        return super().score(X, y, sample_weight=sample_weight, verbose=verbose)

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
        p_pred : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return super().predict_proba(X, verbose=verbose)

    def decision_function(self, X, verbose=0):
        """Predict confidence scores for samples, which may coincide with the
        probability scores in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        scores : array-like of shape (n_samples, n_classes)
            Predicted confidence scores.
        """
        return super().decision_function(X, verbose=verbose)

    def estimate_mi(self, X, y, method=LOG_LOSS_MI_ESTIMATION, **kwargs):
        """Estimate Mutual Information using the specified method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        method : str, optional, default=`LogLossMI`
            The method to use for mutual information estimation. Options include:

            - 'MidPointMI': Estimate MI using Mid-point method.
            - `LogLossMI`: Estimate MI using Log-Loss method.
            - `LogLossMIIsotonicRegression`: Estimate MI using Log-Loss method with Isotonic Regression.
            - `LogLossMIPlattScaling`: Estimate MI using Log-Loss method with Platt Scaling.
            - `LogLossMIBetaCalibration`: Estimate MI using Log-Loss method with Beta Calibration.
            - `LogLossMITemperatureScaling`: Estimate MI using Log-Loss method with Temperature Scaling.
            - `LogLossMIHistogramBinning`: Estimate MI using Log-Loss method with Histogram Binning.
            - `PCSoftmaxMI`: Estimate MI using Softmax probabilities.

        **kwargs : dict, optional
            Additional keyword arguments passed to the estimation methods.

        Returns
        -------
        mutual_information : float
            A mean of estimated MI values from cross-validation splits.
        """
        return super().estimate_mi(X=X, y=y, method=method, **kwargs)
