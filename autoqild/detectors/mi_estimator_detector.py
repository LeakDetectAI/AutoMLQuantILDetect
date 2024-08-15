"""Detects leakage by estimating mutual information using GMM or MINE estimators."""
from .sklearn_leakage_detector import SklearnLeakageDetector
from ..mi_estimators import GMMMIEstimator, MineMIEstimatorMSE
from ..utilities import *

__all__ = ["MIEstimationLeakageDetector"]


class MIEstimationLeakageDetector(SklearnLeakageDetector):
    """
    MIEstimationLeakageDetector class for detecting information leakage using mutual information (MI) estimation techniques.

    This class extends `SklearnLeakageDetector` to detect information leakage in machine learning experiments using mutual information
    estimation techniques. The class supports two primary MI estimation methods: MINE (Mutual Information Neural Estimator) and GMM
    (Gaussian Mixture Model). The selected MI estimation technique is used as the base detector for leakage analysis.

    Parameters
    ----------
    mi_technique : str
        The MI estimation technique to use. Options include:
        - `mine_mi_estimator`: Uses MINE model to estimate mutual information.
        - `gmm_mi_estimator`: Uses GMM model to estimate mutual information.

    padding_name : str
        The name of the padding method used in the experiments to obscure or detect leakage.

    learner_params : dict
        Parameters related to the machine learning models (learners) used in the detection process.

    fit_params : dict
        Parameters passed to the `fit` method during model training.

    hash_value : str
        A unique hash value used to identify and manage result files for a specific experiment.

    cv_iterations : int
        The number of cross-validation iterations to perform during model evaluation.

    n_hypothesis : int
        The number of hypotheses or models to be tested for leakage.

    base_directory : str
        The base directory where result files, logs, and backups are stored.

    search_space : dict
        The hyperparameter search space for Bayesian optimization.

    hp_iters : int
        The number of iterations for hyperparameter optimization.

    n_inner_folds : int
        The number of folds for inner cross-validation during hyperparameter optimization.

    validation_loss : str
        The loss function used to evaluate the performance of models during cross-validation.

    random_state : int or RandomState instance, optional
        Controls the randomness for reproducibility, ensuring consistent results across different runs.

    **kwargs : dict, optional
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If an invalid mutual information technique is specified, or if the detection method is not compatible with the
        selected MI estimator.

    Notes
    -----
    This class supports only the one-sample t-test for the detection method when using mutual information estimation.
    Attempting to use nother detection method will result in a `ValueError` being raised.
    """

    def __init__(self, mi_technique, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, search_space=search_space, hp_iters=hp_iters,
                         n_inner_folds=n_inner_folds, validation_loss=validation_loss, random_state=random_state,
                         **kwargs)

        if mi_technique == MINE_MI_ESTIMATOR:
            self.base_detector = MineMIEstimatorMSE
            self.n_jobs = 1
        elif mi_technique == GMM_MI_ESTIMATOR:
            self.base_detector = GMMMIEstimator
            self.n_jobs = 8
        else:
            raise ValueError(f"Invalid mutual information technique: {mi_technique}")

        if self.detection_method != ESTIMATED_MUTUAL_INFORMATION:
            raise ValueError(
                "Only the one-sample t-test based detection method is compatible with mutual information estimation.")

    def __initialize_objects__(self):
        """
        Initializes the results dictionary for storing metric results.

        This method sets up the internal results dictionary, organizing it by hypothesis models and metrics.
        Each model’s metric scores are prepared for storage, along with the majority voting and random classifier
        baselines.

        Notes
        -----
        This method is intended for internal use only and is automatically called during initialization.
        """
        for i in range(self.n_hypothesis):
            self.results[f"model_{i}"] = {}
            self.results[f"model_{i}"][ESTIMATED_MUTUAL_INFORMATION] = []

    def hyperparameter_optimization(self, X, y):
        """
        Performs Bayesian hyperparameter optimization to identify the best model parameters.

        This method uses a Bayesian search strategy to explore a predefined hyperparameter search space and selects the
        optimal configuration based on the specified validation loss. The method performs cross-validation within the
        search to ensure that the selected hyperparameters generalize well.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be used for training during hyperparameter optimization.

        y : array-like of shape (n_samples,)
            The target values (class labels) corresponding to X.

        Returns
        -------
        int
            The size of the training dataset after reduction (if applicable).

        Raises
        ------
        Exception
            If an error occurs during the Bayesian search fitting process.
        """
        return super().hyperparameter_optimization(X, y)

    def fit(self, X, y):
        """
        Fits the model using cross-validation and performs hyperparameter optimization.

        This method first checks if the model has already been fitted. If not, it runs the hyperparameter optimization process
        followed by cross-validation on the specified number of hypotheses. The model is trained using a stratified split of the
        dataset, and results are evaluated using predefined metrics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data used for training the models.

        y : array-like of shape (n_samples,)
            The target values (class labels) corresponding to X.

        Notes
        -----
        During fitting, random classifier and majority voting classifier performance is also calculated for comparison.
        """
        if self._is_fitted_:
            self.logger.info(f"Model already fitted for the padding {self.padding_code}")
        else:
            train_size = self.hyperparameter_optimization(X, y)
            for i in range(self.n_hypothesis):
                loss, learner_params = self.estimators[i]
                self.logger.info(f"**********  Model {i + 1} with loss {loss} **********")
                self.logger.info(f"Parameters {print_dictionary(learner_params)}")
                model = self.base_detector(**learner_params)
                for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                    train_index = train_index[:train_size]
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model.fit(X=X_train, y=y_train)
                    self.logger.info(f"************************* Split {k + 1} **************************")
                    metric_loss = model.estimate_mi(X, y)
                    self.logger.info(f"Metric {ESTIMATED_MUTUAL_INFORMATION}: Value {metric_loss}")
                    model_name = list(self.results.keys())[i]
                    self.results[model_name][ESTIMATED_MUTUAL_INFORMATION].append(metric_loss)
            self.__store_results__()

    def detect(self):
        """
        Executes the detection process to identify potential information leakage using statistical tests.

        The method applies various statistical techniques, such as paired t-tests and Fisher’s exact test, to detect
        significant differences in model performance that may indicate information leakage. The decision is made based
        on the results of these tests, accounting for multiple hypothesis corrections.

        Returns
        -------
        detection_decision : bool
            Indicates whether any models showed significant leakage.
        hypothesis_rejected : int
            The number of models flagged for leakage.

        Notes
        -----
        The method implements a Holm-Bonferroni correction to control the family-wise error rate for multiple models.
        """
        return super().detect()
