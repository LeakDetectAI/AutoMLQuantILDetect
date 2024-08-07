import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state

from autoqild.bayes_search_hpo.bayes_search_utils import probability_calibration, get_scores
from autoqild.detectors.utils import calibrators, calibrator_params, mi_estimation_metrics
from autoqild.mi_estimators.mi_base_class import MIEstimatorBase
from autoqild.utilities import *


class ClassficationMIEstimator(MIEstimatorBase):

    def __init__(self, base_estimator=RandomForestClassifier, learner_params={}, random_state=None, **kwargs):
        self.base_learner = base_estimator(**learner_params)
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(ClassficationMIEstimator.__name__)

    def fit(self, X, y, **kwd):
        self.base_learner.fit(X, y)

    def predict(self, X, verbose=0):
        return self.base_learner.predict(X)

    def score(self, X, y, sample_weight=None, verbose=0):
        return self.base_learner.score(X, y, sample_weight=sample_weight, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        return self.base_learner.predict_proba(X, verbose=verbose)

    def decision_function(self, X, verbose=0):
        return self.base_learner.decision_function(X, verbose=verbose)

    def estimate_mi(self, X, y, method=LOG_LOSS_MI_ESTIMATION, **kwargs):
        evaluation_metric = mi_estimation_metrics[method]
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.70, random_state=0)
        estimated_mis = []
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p_pred, y_pred = get_scores(X_train, self.base_learner)
            if LOG_LOSS_MI_ESTIMATION in method or PC_SOFTMAX_MI_ESTIMATION in method:
                calibrator_technique = None
                for key in calibrators.keys():
                    if key in method:
                        calibrator_technique = key
                if calibrator_technique is not None:
                    calibrator = calibrators[calibrator_technique]
                    c_params = calibrator_params[calibrator_technique]
                    calibrator = calibrator(**c_params)
                    try:
                        p_pred_cal = probability_calibration(X_train=X_train, y_train=y_train, X_test=X,
                                                             classifier=self.base_learner, calibrator=calibrator)
                        estimated_mi = evaluation_metric(y, p_pred_cal)
                    except Exception as error:
                        log_exception_error(self.logger, error)
                        self.logger.error("Error while calibrating the probabilities estimating mi without calibration")
                        estimated_mi = evaluation_metric(y_train, p_pred)
                else:
                    estimated_mi = evaluation_metric(y_train, p_pred)
            else:
                estimated_mi = evaluation_metric(y_train, y_pred)
            estimated_mis.append(estimated_mi)
        return estimated_mis
