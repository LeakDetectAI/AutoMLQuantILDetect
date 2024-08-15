"""Uses the TabPFN model to detect information leakage, particularly in small
tabular datasets."""
import os

from .sklearn_leakage_detector import SklearnLeakageDetector
from ..automl import AutoTabPFNClassifier
from ..utilities import *

__all__ = ["TabPFNLeakageDetector"]


class TabPFNLeakageDetector(SklearnLeakageDetector):
    """TabPFNLeakageDetector class for detecting information leakage using the
    TabPFN model.

    This class extends `SklearnLeakageDetector` to perform information leakage detection using the TabPFN model, which is particularly
    effective for small tabular datasets. The class incorporates hyperparameter optimization, dataset reduction, and cross-validation,
    making it suitable for scenarios requiring lightweight and efficient models.

    Parameters
    ----------
    padding_name : str
        The name of the padding method used in the experiments to obscure or detect leakage.

    learner_params : dict
        Parameters related to the TabPFN model used in the detection process.

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
    """

    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, search_space=search_space, hp_iters=hp_iters,
                         n_inner_folds=n_inner_folds, validation_loss=validation_loss, random_state=random_state,
                         **kwargs)
        self.n_jobs = 8
        self.base_detector = AutoTabPFNClassifier
        if self.base_detector == AutoTabPFNClassifier:
            self.learner_params["base_path"] = os.path.join(base_directory, OPTIMIZER_FOLDER, hash_value,
                                                            self.padding_code)

    def hyperparameter_optimization(self, X, y):
        train_size = super().hyperparameter_optimization(X, y)
        directory_path = self.learner_params["base_path"]
        try:
            os.rmdir(directory_path)
            self.logger.info(f"The directory `{directory_path}` has been removed.")
        except OSError as e:
            self.logger.error(f"Error: {directory_path} : {e.strerror}")
        return train_size

    def fit(self, X, y):
        super().fit(X, y)

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        super().evaluate_scores(X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train, y_pred=y_pred,
                                p_pred=p_pred, model=model, n_model=n_model)

    def detect(self):
        """Executes the detection process to identify potential information
        leakage using statistical tests.

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
