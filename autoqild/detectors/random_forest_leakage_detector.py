from sklearn.ensemble import RandomForestClassifier

from .sklearn_leakage_detector import SklearnLeakageDetector

__all__ = ['RandomForestLeakageDetector']


class RandomForestLeakageDetector(SklearnLeakageDetector):
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        if "n_classes" in learner_params.keys():
            del learner_params['n_classes']
        if "n_features" in learner_params.keys():
            del learner_params['n_features']
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, search_space=search_space, hp_iters=hp_iters,
                         n_inner_folds=n_inner_folds, validation_loss=validation_loss, random_state=random_state,
                         **kwargs)
        self.n_jobs = 8
        self.base_detector = RandomForestClassifier

    def hyperparameter_optimization(self, X, y):
        return super().hyperparameter_optimization(X, y)

    def fit(self, X, y):
        super().fit(X, y)

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        super().evaluate_scores(X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train, y_pred=y_pred,
                                p_pred=p_pred, model=model, n_model=n_model)

    def detect(self):
        return super().detect()