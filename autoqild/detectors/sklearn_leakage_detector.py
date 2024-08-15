"""A versatile leakage detection class built on top of the scikit-learn framework, supporting multiple estimators."""
import copy
import gc
import logging
import os

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from .ild_base_class import InformationLeakageDetector
from ..automl.tabpfn_classifier import AutoTabPFNClassifier
from ..bayes_search import *
from ..utilities import *


class SklearnLeakageDetector(InformationLeakageDetector):
    """
    SklearnLeakageDetector class for detecting information leakage using a scikit-learn-based model.

    This class extends the `InformationLeakageDetector` base class and incorporates hyperparameter optimization via Bayesian search,
    model fitting, and cross-validation using scikit-learn models. It supports the detection of information leakage in machine learning
    experiments by analyzing the model’s behavior with various padding techniques. The class is highly configurable and works with different
    search spaces, loss functions, and validation strategies.

    Parameters
    ----------
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
        Additional keyword arguments passed to the parent class and used in model fitting.

    Attributes
    ----------
    search_space : dict
        The hyperparameter search space used in Bayesian optimization.

    hp_iters : int
        The number of iterations for hyperparameter optimization.

    n_inner_folds : int
        Number of folds for inner cross-validation.

    validation_loss : str
        The loss function used for validation during hyperparameter tuning.

    inner_cv_iterator : StratifiedShuffleSplit
        Cross-validation iterator used for inner folds during hyperparameter optimization.

    tabpfn_folder : str
        Directory where TabPFN optimization results are saved.

    n_jobs : int
        Number of parallel jobs for hyperparameter search.

    logger : logging.Logger
        Logger instance for recording the process of leakage detection.

    Methods
    -------
    hyperparameter_optimization(X, y)
        Performs Bayesian hyperparameter optimization to identify the best model parameters.

    fit(X, y)
        Fits the model using cross-validation, applying hyperparameter optimization if necessary.

    reduce_dataset(X, y)
        Reduces the dataset size if the number of instances exceeds a threshold, optimizing for lightweight models.

    evaluate_scores(X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model)
        Evaluates the performance of the model using various metrics and stores the results.

    detect()
        Executes the detection process to identify potential information leakage using statistical tests.
    """

    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, random_state=random_state, **kwargs)
        self.search_space = search_space
        self.hp_iters = hp_iters
        self.n_inner_folds = n_inner_folds
        self.validation_loss = validation_loss
        self.inner_cv_iterator = StratifiedShuffleSplit(n_splits=self.n_inner_folds, test_size=0.30,
                                                        random_state=self.random_state)
        self.tabpfn_folder = os.path.join(base_directory, OPTIMIZER_FOLDER, hash_value,
                                          f"{self.padding_code}.pkl")
        create_directory_safely(self.tabpfn_folder, True)
        self.logger = logging.getLogger(SklearnLeakageDetector.__name__)
        self.n_jobs = 10

    def hyperparameter_optimization(self, X, y):
        """
        Performs Bayesian hyperparameter optimization to identify the best model parameters.

        This method uses a Bayesian search strategy to explore a predefined hyperparameter search space and selects the optimal
        configuration based on the specified validation loss. The method performs cross-validation within the search to ensure
        that the selected hyperparameters generalize well.

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
        X_train, y_train = self.__get_training_dataset__(X, y)
        learner = self.base_detector(**self.learner_params)
        bayes_search_params = dict(estimator=learner, search_spaces=self.search_space, n_iter=self.hp_iters,
                                   scoring=self.validation_loss, n_jobs=self.n_jobs, cv=self.inner_cv_iterator,
                                   error_score=0, random_state=self.random_state,
                                   optimizers_file_path=self.tabpfn_folder)
        bayes_search = BayesSearchCV(**bayes_search_params)
        search_keys = list(self.search_space.keys())
        search_keys.sort()
        self.logger.info(f"Search Keys {search_keys}")
        callback = log_callback(search_keys)
        X_train, y_train = self.reduce_dataset(X_train, y_train)
        try:
            bayes_search.fit(X_train, y_train, groups=None, callback=callback, **self.fit_params)
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error(" Cannot fit the Bayes SearchCV ")
        train_size = X_train.shape[0]
        if learner is not None:
            del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.estimators = []
        for i in range(self.n_hypothesis):
            learner_params = copy.deepcopy(self.learner_params)
            loss, learner_params = update_params_at_k(bayes_search, search_keys, learner_params, k=i)
            self.estimators.append([loss, learner_params])
        return train_size

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
                for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                    train_index = train_index[:train_size]
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = self.base_detector(**learner_params)
                    X_train, y_train = self.reduce_dataset(X_train, y_train)
                    X_test, y_test = self.reduce_dataset(X_test, y_test)
                    model.fit(X=X_train, y=y_train)
                    p_pred, y_pred = get_scores(X_test, model)
                    self.logger.info(f"************************* Split {k + 1} **************************")
                    self.evaluate_scores(X_test, X_train, y_test, y_train, y_pred, p_pred, model, i)
                    if i == 0:
                        self.__calculate_random_classifier_accuracy__(X_train, y_train, X_test, y_test)
                        self.__calculate_majority_voting_accuracy__(X_train, y_train, X_test, y_test)
                    directory_path = learner_params.get(`base_path`, None)
                    if directory_path is not None:
                        try:
                            os.rmdir(directory_path)
                            self.logger.info(f"The directory `{directory_path}` has been removed.")
                        except OSError as e:
                            self.logger.error(f"Error: {directory_path} : {e.strerror}")
            self.__store_results__()

    def reduce_dataset(self, X, y):
        """
        Reduces the dataset size for optimization purposes if the number of instances is too large.

        This method is specifically useful for scenarios where lightweight models like TabPFN are being used, and the dataset
        is too large to fit into memory or optimize efficiently. It reduces the dataset size to a maximum threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix.

        y : array-like of shape (n_samples,)
            The target values (class labels) corresponding to X.

        Returns
        -------
        tuple
            Reduced versions of X and y, if applicable.
        """
        if X.shape[0] > 4000 and self.base_detector == AutoTabPFNClassifier:
            reduced_size = 4000
            self.logger.info(f"Initial instances {X.shape[0]} reduced to {reduced_size}")
            X, _, y, _ = train_test_split(X, y, train_size=reduced_size,
                                          stratify=y, random_state=self.random_state)
        return X, y

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        """
        Evaluate and store model performance metrics for the detection process.

        This method computes various evaluation metrics, such as log-loss, accuracy, and confusion matrix, for the model`s
        predictions. It also supports probability calibration using techniques like isotonic regression and Platt scaling.
        The results are stored and logged for further analysis.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The feature matrix for the test set.

        X_train : array-like of shape (n_samples, n_features)
            The feature matrix for the training set.

        y_test : array-like of shape (n_samples,)
            The true target labels for the test data.

        y_train : array-like of shape (n_samples,)
            The true target labels for the training data.

        y_pred : array-like of shape (n_samples,)
            The predicted target labels for the test set.

        p_pred : array-like of shape (n_samples, n_classes)
            The predicted class probabilities for the test data.

        model : object
            The trained model being evaluated.

        n_model : int
            The index of the model in the list of evaluated models.
        """
        super().evaluate_scores(X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train, y_pred=y_pred,
                                p_pred=p_pred, model=model, n_model=n_model)

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

