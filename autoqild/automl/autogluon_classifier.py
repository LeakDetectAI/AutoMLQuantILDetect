import logging
import os.path
import shutil

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from pycilt.automl.automl_core import AutomlClassifier
from pycilt.automl.model_configurations import hyperparameters, reduced_hyperparameters
from pycilt.utils import log_exception_error
from sklearn.utils import check_random_state


class AutoGluonClassifier(AutomlClassifier):

    def __init__(self, n_features, n_classes, time_limit=1800, output_folder=None, eval_metric='accuracy',
                 use_hyperparameters=True, delete_tmp_folder_after_terminate=True, auto_stack=True,
                 remove_boosting_models=True, verbosity=6, random_state=None, **kwargs):
        self.logger = logging.getLogger(name=AutoGluonClassifier.__name__)
        self.random_state = check_random_state(random_state)
        self.output_folder = output_folder
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.hyperparameter_tune_kwargs = {'scheduler': 'local', 'searcher': 'auto'}
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
            self.exclude_model_types = ['GBM', 'CAT', 'XGB', 'LGB', 'KNN', 'NN_TORCH', "AG_AUTOMM", 'LR']
        else:
            self.exclude_model_types = ["AG_AUTOMM", 'LR']
        self.auto_stack = auto_stack
        self.n_features = n_features
        self.n_classes = n_classes
        self.sample_weight = "auto_weight"
        self.time_limit = time_limit
        self.model = None
        self.class_label = 'class'
        self.columns = [f'feature_{i}' for i in range(self.n_features)] + [self.class_label]
        if self.n_classes > 2:
            self.problem_type = 'multiclass'
        if self.n_classes == 2:
            self.problem_type = 'binary'
        self.leaderboard = None
        #        if "pc2" in os.environ["HOME"]:
        #            tmp_dir_path = os.path.join(os.environ["PFS_FOLDER"], "tmp")
        #            if not os.path.isdir(tmp_dir_path):
        #                os.mkdir(tmp_dir_path)
        #            os.environ['RAY_LOG_DIR'] = os.environ['RAY_HOME'] = os.environ['TMPDIR'] = tmp_dir_path

    @property
    def _is_fitted_(self) -> bool:
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
            time_taken = self.leaderboard['fit_time'].sum() + self.leaderboard['pred_time_val'].sum() + 20
            difference = self.time_limit - time_taken
            if 200 <= self.time_limit < 300:
                limit = 150
            elif self.time_limit >= 3000:
                limit = 2000
            else:
                limit = 200
            self.logger.info(f"Fitting time of the model {time_taken} and remaining {difference}, limit {limit}")
            num_models = len(self.leaderboard['fit_time'])
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
        # Set the alarm to trigger after the specified time limit
        self.logger.info("Fitting Started")
        train_data = self.convert_to_dataframe(X, y)
        while not self._is_fitted_:
            try:
                self.logger.info("Fitting the model from the scratch")
                self.model = TabularPredictor(label=self.class_label, sample_weight=self.sample_weight,
                                              problem_type=self.problem_type, eval_metric=self.eval_metric,
                                              path=self.output_folder, verbosity=self.verbosity)
                self.model.fit(train_data, time_limit=self.time_limit, hyperparameters=hyperparameters,
                               hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs, auto_stack=self.auto_stack,
                               excluded_model_types=self.exclude_model_types)
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error("Fit function did not work, checking the saved models")
        self.leaderboard = self.model.leaderboard(extra_info=True)
        if self.delete_tmp_folder_after_terminate:
            self.model.delete_models(models_to_keep='best', dry_run=False)
            self.model.save_space()

    def predict(self, X, verbose=0):
        test_data = self.convert_to_dataframe(X, None)
        y_pred = self.model.predict(test_data)
        return y_pred.values

    def score(self, X, y, sample_weight=None, verbose=0):
        test_data = self.convert_to_dataframe(X, y)
        score = self.model.evaluate(test_data)['balanced_accuracy']
        return score

    def predict_proba(self, X, verbose=0):
        test_data = self.convert_to_dataframe(X, None)
        y_pred = self.model.predict_proba(test_data)
        return y_pred.values

    def decision_function(self, X, verbose=0):
        test_data = self.convert_to_dataframe(X, None)
        y_pred = self.model.predict_proba(test_data)
        return y_pred.values

    def convert_to_dataframe(self, X, y=None):
        # Ensure X and y are NumPy arrays
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        # If y is None, generate random labels
        else:
            n_instances = X.shape[0]
            y = self.random_state.choice(self.n_classes, size=n_instances)

        # Ensure X and y are writeable
        X = np.copy(X)
        X.flags.writeable = True
        y = np.copy(y)
        y.flags.writeable = True

        # Concatenate X and y
        data = np.concatenate((X, y[:, None]), axis=1)

        # Check the number of features
        if self.n_features != X.shape[-1]:
            raise ValueError(f"Dataset passed does not contain {self.n_features} features")

        # Convert to DataFrame
        df_data = pd.DataFrame(data=data, columns=self.columns)
        return df_data

    def get_k_rank_model(self, k):
        self.leaderboard.sort_values(['score_val'], ascending=False, inplace=True)
        model_name = self.leaderboard.iloc[k - 1]['model']
        model = self.model._trainer.load_model(model_name)
        return model

    def get_model(self, model_name):
        self.leaderboard.sort_values(['score_val'], ascending=False, inplace=True)
        # model_name = self.leaderboard.iloc[k - 1]['model']
        model = self.model._trainer.load_model(model_name)
        return model
