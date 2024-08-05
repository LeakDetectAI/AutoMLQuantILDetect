from .sklearn_leakage_detector import SklearnLeakageDetector
from ..constants import *
from ..mi_estimators import GMMMIEstimator, MineMIEstimatorHPO
from ..utils import print_dictionary


class MIEstimationLeakageDetector(SklearnLeakageDetector):
    def __init__(self, mi_technique, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, search_space=search_space, hp_iters=hp_iters,
                         n_inner_folds=n_inner_folds, validation_loss=validation_loss, random_state=random_state,
                         **kwargs)

        if mi_technique == MINE_MI_ESTIMATOR:
            self.base_detector = MineMIEstimatorHPO
            self.n_jobs = 1
        if mi_technique == GMM_MI_ESTIMATOR:
            self.base_detector = GMMMIEstimator
            self.n_jobs = 8

    def __initialize_objects__(self):
        for i in range(self.n_hypothesis):
            self.results[f'model_{i}'] = {}
            self.results[f'model_{i}'][ESTIMATED_MUTUAL_INFORMATION] = []

    def perform_hyperparameter_optimization(self, X, y):
        return super().perform_hyperparameter_optimization(X, y)

    def fit(self, X, y):
        if self._is_fitted_:
            self.logger.info(f"Model already fitted for the padding {self.padding_code}")
        else:
            train_size = self.perform_hyperparameter_optimization(X, y)
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
            self.store_results()
