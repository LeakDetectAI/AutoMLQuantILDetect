from autoqild.automl import AutoTabPFNClassifier
from autoqild.core.mi_estimator_classification import ClassficationMIEstimator
from autoqild.utilities.constants import LOG_LOSS_MI_ESTIMATION


class AutoGluonEstimator(ClassficationMIEstimator):

    def __init__(self, n_features, n_classes, n_ensembles=100, n_reduced=20, reduction_technique='select_from_model_rf',
                 base_path=None, random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.base_estimator = AutoTabPFNClassifier
        self.learner_params = dict(n_features=n_features, n_classes=n_classes, n_ensembles=n_ensembles,
                                   n_reduced=n_reduced, reduction_technique=reduction_technique,
                                   base_path=base_path, random_state=random_state)
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
