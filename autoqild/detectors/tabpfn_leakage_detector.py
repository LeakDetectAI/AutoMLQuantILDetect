import os

from .sklearn_leakage_detector import SklearnLeakageDetector
from ..automl import AutoTabPFNClassifier
from ..utilities import *

__all__ = ['TabPFNLeakageDetector']


class TabPFNLeakageDetector(SklearnLeakageDetector):
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
            self.learner_params['base_path'] = os.path.join(base_directory, OPTIMIZER_FOLDER, hash_value,
                                                            self.padding_code)

    def hyperparameter_optimization(self, X, y):
        train_size = super().hyperparameter_optimization(X, y)
        directory_path = self.learner_params['base_path']
        try:
            os.rmdir(directory_path)
            self.logger.info(f"The directory '{directory_path}' has been removed.")
        except OSError as e:
            self.logger.error(f"Error: {directory_path} : {e.strerror}")
        return train_size

    def fit(self, X, y):
        super().fit(X, y)

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        super().evaluate_scores(X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train, y_pred=y_pred,
                                p_pred=p_pred, model=model, n_model=n_model)

    def detect(self):
        return super().detect()