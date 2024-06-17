import logging

import numpy as np
from pycilt.bayes_search_utils import get_scores
from pycilt.utils import normalize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


class BayesPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, dataset_obj, random_state=None, **kwargs):
        self.dataset_obj = dataset_obj
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(BayesPredictor.__name__)
        self.n_classes = None

    def fit(self, X, y, **kwd):
        self.n_classes = len(np.unique(y))
        pass

    def predict(self, X, verbose=0):
        pred_probabilities = self.predict_proba(X=X, verbose=verbose)
        y_pred = pred_probabilities.argmax(axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.predict(X)
        acc_bp = np.mean(y_pred == y)
        return acc_bp

    def decision_function(self, X, verbose=0):
        prob_predictions = np.zeros((X.shape[0], self.dataset_obj.n_classes))
        for k_class in self.dataset_obj.class_labels:
            if self.dataset_obj.flip_y == 0.0:
                prob_predictions[:, k_class] = self.dataset_obj.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                prob_predictions[:, k_class] = self.dataset_obj.get_prob_flip_y_given_x(X=X, class_label=k_class)
        if self.n_classes == 2:
            prob_predictions = prob_predictions[:, 1]
        return prob_predictions

    def predict_proba(self, X, verbose=0):
        prob_predictions = np.zeros((X.shape[0], self.dataset_obj.n_classes))
        for k_class in self.dataset_obj.class_labels:
            if self.dataset_obj.flip_y == 0.0:
                prob_predictions[:, k_class] = self.dataset_obj.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                prob_predictions[:, k_class] = self.dataset_obj.get_prob_flip_y_given_x(X=X, class_label=k_class)
        prob_predictions = normalize(prob_predictions, axis=1)
        return prob_predictions

    def get_bayes_predictor_scores(self):
        max_acc = -np.inf
        y_true = None
        y_pred = None
        p_pred = None
        for i in range(100):
            X, y = self.dataset_obj.generate_dataset()
            pred = self.predict(X)
            acc_bp = np.mean(pred == y)
            if acc_bp > max_acc:
                self.logger.info(f"Accuracy of Bayes Predictor is {acc_bp}")
                max_acc = acc_bp
                y_true = np.copy(y)
                p_pred, y_pred = get_scores(X, self)
        return y_true, y_pred, p_pred

    def decision_function(self, X, verbose=0):
        prob_predictions = self.predict_proba(X=X, verbose=verbose)
        return prob_predictions
