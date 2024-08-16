"""A leakage detection class leveraging AutoGluon for hyperparameter
optimization and model evaluation."""

import logging
import os.path

from .ild_base_class import InformationLeakageDetector
from ..automl import AutoGluonClassifier
from ..bayes_search.bayes_search_utils import get_scores
from ..utilities import *

__all__ = ["AutoGluonLeakageDetector"]


class AutoGluonLeakageDetector(InformationLeakageDetector):
    """AutoGluonLeakageDetector leverages the AutoGluon framework for detecting
    information leakage in machine learning models. This class extends the
    `InformationLeakageDetector` base class and uses AutoGluon for
    hyperparameter optimization and model training. It evaluates potential
    information leakage using various metrics across different cross-validation
    splits.

    Parameters
    ----------
    padding_name : str
        The name of the padding method used in experiments to potentially obscure or prevent leakage.

    learner_params : dict
        Parameters related to the AutoGluon classifier used in the leakage detection process.

    fit_params : dict
        Parameters passed to the `fit` method of the AutoGluon models during training.

    hash_value : str
        A unique hash value used to identify and manage result files for a specific experiment.

    cv_iterations : int
        The number of cross-validation iterations to perform during model evaluation.

    n_hypothesis : int
        The number of hypotheses or models to be tested for leakage.

    base_directory : str
        The base directory where result files, logs, and backups are stored.

    validation_loss : str
        The evaluation metric used to assess model performance during hyperparameter optimization.

    random_state : int or None, optional
        Controls the randomness for reproducibility, ensuring consistent results across different runs.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `InformationLeakageDetector` base class.

    Attributes
    ----------
    base_detector : AutoGluonClassifier
        The base AutoGluon classifier used for model training.

    learner : AutoGluonClassifier instance
        The AutoGluon classifier instance used for the current experiment.

    logger : logging.Logger
        Logger instance used for recording the steps and processes of the leakage detection.
    """

    def __init__(
        self,
        padding_name,
        learner_params,
        fit_params,
        hash_value,
        cv_iterations,
        n_hypothesis,
        base_directory,
        validation_loss,
        random_state=None,
        **kwargs,
    ):
        super().__init__(
            padding_name=padding_name,
            learner_params=learner_params,
            fit_params=fit_params,
            hash_value=hash_value,
            cv_iterations=cv_iterations,
            n_hypothesis=n_hypothesis,
            base_directory=base_directory,
            random_state=random_state,
            **kwargs,
        )
        self.base_detector = AutoGluonClassifier
        self.learner = None
        output_folder = os.path.join(
            base_directory,
            OPTIMIZER_FOLDER,
            hash_value,
            f"{self.padding_code}_autogluon",
        )
        create_directory_safely(output_folder)
        self.learner_params["output_folder"] = output_folder
        self.learner_params["eval_metric"] = validation_loss
        self.learner_params["delete_tmp_folder_after_terminate"] = False
        self.learner_params["remove_boosting_models"] = True
        self.logger = logging.getLogger(AutoGluonLeakageDetector.__name__)

    def hyperparameter_optimization(self, X, y):
        """Performs hyperparameter optimization using AutoGluon to find the
        best models for leakage detection.

        This method runs a Bayesian optimization process to identify the best models according to the specified evaluation metric.
        The optimized models are then stored for subsequent evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix used for training during hyperparameter optimization.

        y : array-like of shape (n_samples,)
            The target values (class labels) corresponding to each row in X.

        Returns
        -------
        int
            The size of the training dataset after the reduction (if applicable).
        """
        X_train, y_train = self.__get_training_dataset__(X, y)
        self.learner = self.base_detector(**self.learner_params)
        self.learner.fit(X_train, y_train)
        for i in range(self.n_hypothesis * 3):
            self.logger.info(f"Getting model at {i}")
            model = self.learner.get_k_rank_model(i + 1)
            self.estimators.append(model)
        train_size = X_train.shape[0]
        return train_size

    def fit(self, X, y, **kwargs):
        """Fits the models using cross-validation and evaluates them for
        information leakage.

        This method performs cross-validation, training the AutoGluon models across different data splits.
        The models are then evaluated for potential leakage using metrics such as accuracy and log-loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input feature matrix used for model training.

        y : array-like of shape (n_samples,)
            The target values (class labels) corresponding to each row in X.
        """
        if self._is_fitted_:
            self.logger.info(f"Model already fitted for the padding {self.padding_code}")
        else:
            train_size = self.hyperparameter_optimization(X, y)
            n_hypothesis = 0
            for i, model in enumerate(self.estimators):
                if n_hypothesis == self.n_hypothesis:
                    break
                try:
                    self.logger.info(
                        f"************** Model {i + 1}: {model.__class__.__name__} **************"
                    )
                    for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                        self.logger.info(
                            f"************************** Split {k + 1} ***************************"
                        )
                        train_index = train_index[:train_size]
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        if i == 0:
                            self.__calculate_random_classifier_accuracy__(
                                X_train, y_train, X_test, y_test
                            )
                            self.__calculate_majority_voting_accuracy__(
                                X_train, y_train, X_test, y_test
                            )
                        train_data = self.learner.convert_to_dataframe(X_train, y_train)
                        test_data = self.learner.convert_to_dataframe(X_test, None)
                        X_t = train_data.drop(columns=["class"])
                        y_t = train_data["class"]
                        model._n_repeats_finished = 0
                        n_repeat_start = 0
                        model.fit(X=X_t, y=y_t, n_repeat_start=n_repeat_start)
                        p_pred, y_pred = get_scores(test_data, model)
                        self.evaluate_scores(
                            X_test,
                            X_train,
                            y_test,
                            y_train,
                            y_pred,
                            p_pred,
                            model,
                            n_hypothesis,
                        )
                    n_hypothesis += 1
                    self.logger.info(f"Hypothesis Done {n_hypothesis} out of {self.n_hypothesis}")
                except Exception as error:
                    log_exception_error(self.logger, error)
                    self.logger.error(f"Problem with fitting the model")
            self.__store_results__()

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        """Evaluates and stores model performance metrics for the detection
        process.

        This method computes various evaluation metrics, such as log-loss, accuracy, and confusion matrix, for the
        model`s predictions. The results are stored and logged for further analysis.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input feature matrix for the test set.

        X_train : array-like of shape (n_samples, n_features)
            The input feature matrix for the training set.

        y_test : array-like of shape (n_samples,)
            The true target labels for the test set.

        y_train : array-like of shape (n_samples,)
            The true target labels for the training set.

        y_pred : array-like of shape (n_samples,)
            The predicted labels for the test set.

        p_pred : array-like of shape (n_samples, n_classes)
            The predicted class probabilities for the test set.

        model : object
            The trained model that is being evaluated.

        n_model : int
            The index of the model within the list of models being evaluated.
        """
        super().evaluate_scores(
            X_test=X_test,
            X_train=X_train,
            y_test=y_test,
            y_train=y_train,
            y_pred=y_pred,
            p_pred=p_pred,
            model=model,
            n_model=n_model,
        )

    def detect(self, detection_method=LOG_LOSS_MI_ESTIMATION):
        """Executes the detection process to identify potential information
        leakage using the specified method.

        Parameters
        ----------
        detection_method : str
        The method to use for detecting information leakage. Options include:
        - `paired-t-test`: Uses paired t-test to compare the accuracy of models against the majority voting baseline.
        - `paired-t-test-random`: Uses paired t-test to compare the accuracy of models against a random classifier.
        - `fishers-exact-mean`: Applies Fisher's Exact Test on the confusion matrix and computes the mean p-value.
        - `fishers-exact-median`: Applies Fisher's Exact Test on the confusion matrix and computes the median p-value.
        - `mid_point_mi`: Detects leakage using the midpoint mutual information estimation.
        - `log_loss_mi`: Detects leakage using log loss mutual information estimation.
        - `log_loss_mi_isotonic_regression`: Uses log loss mutual information estimation with isotonic regression calibration.
        - `log_loss_mi_platt_scaling`: Uses log loss mutual information estimation with Platt scaling calibration.
        - `log_loss_mi_beta_calibration`: Uses log loss mutual information estimation with beta calibration.
        - `log_loss_mi_temperature_scaling`: Uses log loss mutual information estimation with temperature scaling.
        - `log_loss_mi_histogram_binning`: Uses log loss mutual information estimation with histogram binning.
        - `p_c_softmax_mi`: Uses PC-Softmax mutual information estimation for detection.

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
        return super().detect(detection_method=detection_method)
