"""Uses a Multi-Layer Perceptron (MLP) for detecting leakage using deep
learning approaches."""

from .sklearn_leakage_detector import SklearnLeakageDetector
from ..utilities import LOG_LOSS_MI_ESTIMATION
from ..classifiers import MultiLayerPerceptron

__all__ = ["MLPLeakageDetector"]


class MLPLeakageDetector(SklearnLeakageDetector):
    """MLPLeakageDetector class for detecting information leakage using a
    multi-layer perceptron (MLP) model.

    This class extends `SklearnLeakageDetector` to analyze information leakage using a multi-layer perceptron (MLP) as the base model.
    The class is designed for experiments where deep learning models are employed for leakage detection. It integrates hyperparameter
    optimization and cross-validation to improve detection accuracy.

    Parameters
    ----------
    padding_name : str
        The name of the padding method used in the experiments to obscure or detect leakage.

    learner_params : dict
        Parameters related to the MLP model used in the detection process.

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

    def __init__(
        self,
        padding_name,
        learner_params,
        fit_params,
        hash_value,
        cv_iterations,
        n_hypothesis,
        base_directory,
        search_space,
        hp_iters,
        n_inner_folds,
        validation_loss,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            padding_name=padding_name,
            learner_params=learner_params,
            fit_params=fit_params,
            hash_value=hash_value,
            cv_iterations=cv_iterations,
            n_hypothesis=n_hypothesis,
            base_directory=base_directory,
            search_space=search_space,
            hp_iters=hp_iters,
            n_inner_folds=n_inner_folds,
            validation_loss=validation_loss,
            random_state=random_state,
            **kwargs
        )
        self.n_jobs = 1
        self.base_detector = MultiLayerPerceptron

    def hyperparameter_optimization(self, X, y):
        """Performs Bayesian hyperparameter optimization to identify the best
        model parameters.

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
        return super().hyperparameter_optimization(X, y)

    def fit(self, X, y):
        """Fits the model using cross-validation and performs hyperparameter
        optimization.

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
        super().fit(X, y)

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        """Evaluate and store model performance metrics for the detection
        process.

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

    def detect(self, detection_method="log_loss_mi"):
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
