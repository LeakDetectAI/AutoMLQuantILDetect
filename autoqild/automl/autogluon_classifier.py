"AutoGluonClassifier is a wrapper for building, training, and evaluating an AutoML model using AutoGluon."
import logging
import os.path
import shutil

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.utils import check_random_state

from autoqild.automl.automl_core import AutomlClassifier
from .model_configurations import hyperparameters, reduced_hyperparameters
from ..utilities.utils import log_exception_error


class AutoGluonClassifier(AutomlClassifier):
    """AutoGluonClassifier is a wrapper for building, training, and evaluating
    an AutoML model using AutoGluon.

    This class facilitates the use of AutoGluon for automatic machine learning (AutoML) tasks,
    specifically focusing on classification problems. It handles various aspects of model training,
    including hyperparameter tuning, model stacking, and model evaluation. The class is designed to
    work seamlessly with the AutoGluon library, allowing users to leverage its powerful features with
    minimal setup.

    Parameters
    ----------
    n_features : int
        Number of features or dimensionality of the input data.
    n_classes : int
        Number of classes in the classification problem.
    time_limit : int, optional
        Time limit for training the model, in seconds. Default is 1800.
    output_folder : str, optional
        Path to the directory where the trained model and related files will be saved. Default is None.
    eval_metric : str, optional
        Evaluation metric used to assess the performance of the model. Default is `accuracy`.
    use_hyperparameters : bool, optional
        Flag indicating whether to use predefined hyperparameters for model training. Default is True.
    delete_tmp_folder_after_terminate : bool, optional
        Flag indicating whether to delete the temporary folder after model training is complete. Default is True.
    auto_stack : bool, optional
        Flag indicating whether to use automatic stacking of models in AutoGluon. Default is True.
    remove_boosting_models : bool, optional
        Flag indicating whether to exclude boosting models (like GBM, CAT, XGB) from the hyperparameters. Default is True.
    verbosity : int, optional
        Level of verbosity for logging and output. Default is 6.
    random_state : int or None, optional
        Seed for random number generation to ensure reproducibility. Default is None.

    Attributes
    ----------
    logger : logging.Logger
        Logger object used for logging messages and errors.
    random_state : np.random.RandomState
        Random state instance for reproducibility.
    output_folder : str
        Path to the directory where the trained model and related files will be saved.
    delete_tmp_folder_after_terminate : bool
        Flag indicating whether to delete the temporary folder after model training is complete.
    hyperparameter_tune_kwargs : dict
        Dictionary containing options for hyperparameter tuning, including the scheduler and searcher.
    eval_metric : str
        Evaluation metric used to assess the performance of the model.
    use_hyperparameters : bool
        Flag indicating whether to use predefined hyperparameters for model training.
    verbosity : int
        Level of verbosity for logging and output.
    hyperparameters : dict or None
        Dictionary of hyperparameters used for model training. If `use_hyperparameters` is False, this is None.
    exclude_model_types : list
        List of model types to exclude from the training process.
    auto_stack : bool
        Flag indicating whether to use automatic stacking of models in AutoGluon.
    n_features : int
        Number of features or dimensionality of the input data.
    n_classes : int
        Number of classes in the classification problem.
    sample_weight : str
        Method for determining sample weights during training, default is `auto_weight`.
    time_limit : int
        Time limit for training the model, in seconds.
    model : autogluon.tabular.TabularPredictor or None
        The AutoGluon model object, initialized after fitting.
    class_label : str
        Name of the target label column.
    columns : list
        List of column names for the input DataFrame, including feature names and the class label.
    leaderboard : pandas.DataFrame or None
        DataFrame containing information about the models trained during the fitting process.

    Private Methods
    ---------------
    _is_fitted_() -> bool
        Property to check if the model is already fitted.
    """

    def __init__(self, n_features, n_classes, time_limit=1800, output_folder=None, eval_metric="accuracy",
                 use_hyperparameters=True, delete_tmp_folder_after_terminate=True, auto_stack=True,
                 remove_boosting_models=True, verbosity=6, random_state=None, **kwargs):
        self.logger = logging.getLogger(name=AutoGluonClassifier.__name__)
        self.random_state = check_random_state(random_state)
        self.output_folder = output_folder
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "auto"}
        self.eval_metric = eval_metric
        self.use_hyperparameters = use_hyperparameters
        self.verbosity = verbosity
        if self.use_hyperparameters:
            if remove_boosting_models:
                self.hyperparameters = hyperparameters
            else:
                self.hyperparameters = reduced_hyperparameters
        else:
            self.hyperparameters = None
        if remove_boosting_models:
            self.exclude_model_types = ["GBM", "CAT", "XGB", "LGB", "KNN", "NN_TORCH", "AG_AUTOMM", "LR"]
        else:
            self.exclude_model_types = ["AG_AUTOMM", "LR"]
        self.auto_stack = auto_stack
        self.n_features = n_features
        self.n_classes = n_classes
        self.sample_weight = "auto_weight"
        self.time_limit = time_limit
        self.model = None
        self.class_label = "class"
        self.columns = [f"feature_{i}" for i in range(self.n_features)] + [self.class_label]
        if self.n_classes > 2:
            self.problem_type = "multiclass"
        if self.n_classes == 2:
            self.problem_type = "binary"
        self.leaderboard = None

    @property
    def _is_fitted_(self) -> bool:
        """Check if the model is already fitted.

        Returns
        -------
        _is_fitted_ : bool
            True if the model is fitted, False otherwise.
        """
        basename = os.path.basename(self.output_folder)
        if os.path.exists(self.output_folder):
            try:
                self.model = TabularPredictor.load(self.output_folder)
                self.logger.info(f"Loading the model at {basename}")
                self.leaderboard = self.model.leaderboard(extra_info=True)
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Cannot load the trained model at {basename}")
                self.model = None

        if self.model is not None:
            self.leaderboard = self.model.leaderboard(extra_info=True)
            time_taken = self.leaderboard["fit_time"].sum() + self.leaderboard["pred_time_val"].sum() + 20
            difference = self.time_limit - time_taken
            if 200 <= self.time_limit < 300:
                limit = 150
            elif self.time_limit >= 3000:
                limit = 2000
            else:
                limit = 200
            self.logger.info(f"Fitting time of the model {time_taken} and remaining {difference}, limit {limit}")
            num_models = len(self.leaderboard["fit_time"])
            self.logger.info(f"Number of models trained is {num_models} ")
            if num_models < 1200:
                if num_models <= 50:
                    self.model = None
                    self.logger.info(f"Retraining the model since they are less than 50")
                if difference >= limit:
                    self.model = None
            else:
                self.logger.info("Enough models trained")

        if self.model is None:
            try:
                shutil.rmtree(self.output_folder)
                self.logger.error(f"Since the model is not completely fitted, the folder '{basename}' "
                                  f"and its contents are deleted successfully.")
            except OSError as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Folder does not exist")
        return self.model is not None

    def fit(self, X, y, **kwd):
        """Fit the AutoGluon model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        **kwd : dict, optional
            Additional keyword arguments.
        """
        self.logger.info("Fitting Started")
        train_data = self.convert_to_dataframe(X, y)
        while not self._is_fitted_:
            try:
                self.logger.info("Fitting the model from scratch")
                self.model = TabularPredictor(label=self.class_label, sample_weight=self.sample_weight,
                                              problem_type=self.problem_type, eval_metric=self.eval_metric,
                                              path=self.output_folder, verbosity=self.verbosity)
                self.model.fit(train_data, time_limit=self.time_limit, hyperparameters=self.hyperparameters,
                               hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs, auto_stack=self.auto_stack,
                               excluded_model_types=self.exclude_model_types)
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error("Fit function did not work, checking the saved models")
        self.leaderboard = self.model.leaderboard(extra_info=True)
        if self.delete_tmp_folder_after_terminate:
            self.model.delete_models(models_to_keep="best", dry_run=False)
            self.model.save_space()

    def predict(self, X, verbose=0):
        """Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        test_data = self.convert_to_dataframe(X, None)
        y_pred = self.model.predict(test_data)
        return y_pred.values

    def score(self, X, y, sample_weight=None, verbose=0):
        """Compute the balanced accuracy score for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            True labels.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            Balanced accuracy score.
        """
        test_data = self.convert_to_dataframe(X, y)
        score = self.model.evaluate(test_data)["balanced_accuracy"]
        return score

    def predict_proba(self, X, verbose=0):
        """Predict class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        test_data = self.convert_to_dataframe(X, None)
        y_pred = self.model.predict_proba(test_data)
        return y_pred.values

    def decision_function(self, X, verbose=0):
        """Compute the decision function in form of class probabilities for the
        input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values.
        """
        test_data = self.convert_to_dataframe(X, None)
        y_pred = self.model.predict_proba(test_data)
        return y_pred.values

    def convert_to_dataframe(self, X, y=None):
        """Convert the input data to a DataFrame.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,), optional
            Target vector.

        Returns
        -------
        df_data : pandas.DataFrame
            DataFrame containing the input data.
        """
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        else:
            n_instances = X.shape[0]
            y = self.random_state.choice(self.n_classes, size=n_instances)

        X = np.copy(X)
        X.flags.writeable = True
        y = np.copy(y)
        y.flags.writeable = True

        data = np.concatenate((X, y[:, None]), axis=1)

        if self.n_features != X.shape[-1]:
            raise ValueError(f"Dataset passed does not contain {self.n_features} features")

        df_data = pd.DataFrame(data=data, columns=self.columns)
        return df_data

    def get_k_rank_model(self, k):
        """Get the k-th ranked model from the leaderboard.

        Parameters
        ----------
        k : int
            Rank of the model to retrieve.

        Returns
        -------
        model : autogluon.tabular.TabularPredictor
            The k-th ranked model.
        """
        self.leaderboard.sort_values(["score_val"], ascending=False, inplace=True)
        model_name = self.leaderboard.iloc[k - 1]["model"]
        model = self.model._trainer.load_model(model_name)
        return model

    def get_model(self, model_name):
        """Get a model by its name from the leaderboard.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve.

        Returns
        -------
        model : autogluon.tabular.TabularPredictor
            The specified model.
        """
        self.leaderboard.sort_values(["score_val"], ascending=False, inplace=True)
        model = self.model._trainer.load_model(model_name)
        return model
