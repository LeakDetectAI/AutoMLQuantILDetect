from autoqild.utilities.constants import LOG_LOSS_MI_ESTIMATION
from autoqild.automl import AutoGluonClassifier
from autoqild.mi_estimators.mi_estimator_classification import ClassficationMIEstimator


class AutoGluonEstimator(ClassficationMIEstimator):

    def __init__(self, n_features, n_classes, time_limit=1800, output_folder=None, eval_metric='accuracy',
                 use_hyperparameters=True, delete_tmp_folder_after_terminate=True, auto_stack=True,
                 remove_boosting_models=True, verbosity=6, random_state=None, **kwargs):
        self.base_estimator = AutoGluonClassifier
        self.learner_params = dict(n_features=n_features, n_classes=n_classes, time_limit=time_limit,
                                   output_folder=output_folder, eval_metric=eval_metric,
                                   use_hyperparameters=use_hyperparameters,
                                   delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
                                   auto_stack=auto_stack, remove_boosting_models=remove_boosting_models,
                                   verbosity=verbosity, random_state=random_state)
        self.base_learner = self.base_estimator(**self.learner_params)

    def fit(self, X, y, **kwd):
        super().fit(X, y, **kwd)

    def predict(self, X, verbose=0):
        return super().predict(X, verbose=verbose)

    def score(self, X, y, sample_weight=None, verbose=0):
        return super().score(X, y, sample_weight=sample_weight, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        return super().predict_proba(X, verbose=verbose)

    def decision_function(self, X, verbose=0):
        return super().decision_function(X, verbose=verbose)

    def estimate_mi(self, X, y, method=LOG_LOSS_MI_ESTIMATION, **kwargs):
        return super().estimate_mi(X=X, y=y, method=method, **kwargs)