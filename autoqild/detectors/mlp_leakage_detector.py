from .sklearn_leakage_detector import SklearnLeakageDetector
from ..classifiers import MultiLayerPerceptron


class MLPLeakageDetector(SklearnLeakageDetector):
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, search_space=search_space, hp_iters=hp_iters,
                         n_inner_folds=n_inner_folds, validation_loss=validation_loss, random_state=random_state,
                         **kwargs)
        self.n_jobs = 1
        self.base_detector = MultiLayerPerceptron

    def perform_hyperparameter_optimization(self, X, y):
        return super().perform_hyperparameter_optimization(X, y)

    def fit(self, X, y):
        super().fit(X, y)
