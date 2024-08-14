import copy
import gc
import logging
import os

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from .ild_base_class import InformationLeakageDetector
from ..automl.tabpfn_classifier import AutoTabPFNClassifier
from ..bayes_search import *
from ..utilities import *


class SklearnLeakageDetector(InformationLeakageDetector):
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, random_state=random_state, **kwargs)
        self.search_space = search_space
        self.hp_iters = hp_iters
        self.n_inner_folds = n_inner_folds
        self.validation_loss = validation_loss
        self.inner_cv_iterator = StratifiedShuffleSplit(n_splits=self.n_inner_folds, test_size=0.30,
                                                        random_state=self.random_state)
        self.tabpfn_folder = os.path.join(base_directory, OPTIMIZER_FOLDER, hash_value,
                                          f"{self.padding_code}.pkl")
        create_directory_safely(self.tabpfn_folder, True)
        self.logger = logging.getLogger(SklearnLeakageDetector.__name__)
        self.n_jobs = 10

    def perform_hyperparameter_optimization(self, X, y):
        X_train, y_train = self.get_training_dataset(X, y)
        learner = self.base_detector(**self.learner_params)
        bayes_search_params = dict(estimator=learner, search_spaces=self.search_space, n_iter=self.hp_iters,
                                   scoring=self.validation_loss, n_jobs=self.n_jobs, cv=self.inner_cv_iterator,
                                   error_score=0, random_state=self.random_state,
                                   optimizers_file_path=self.tabpfn_folder)
        bayes_search = BayesSearchCV(**bayes_search_params)
        search_keys = list(self.search_space.keys())
        search_keys.sort()
        self.logger.info(f"Search Keys {search_keys}")
        callback = log_callback(search_keys)
        X_train, y_train = self.reduce_dataset(X_train, y_train)
        try:
            bayes_search.fit(X_train, y_train, groups=None, callback=callback, **self.fit_params)
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error(" Cannot fit the Bayes SearchCV ")
        train_size = X_train.shape[0]
        if learner is not None:
            del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.estimators = []
        for i in range(self.n_hypothesis):
            learner_params = copy.deepcopy(self.learner_params)
            loss, learner_params = update_params_at_k(bayes_search, search_keys, learner_params, k=i)
            self.estimators.append([loss, learner_params])
        return train_size

    def fit(self, X, y):
        if self._is_fitted_:
            self.logger.info(f"Model already fitted for the padding {self.padding_code}")
        else:
            train_size = self.perform_hyperparameter_optimization(X, y)
            for i in range(self.n_hypothesis):
                loss, learner_params = self.estimators[i]
                self.logger.info(f"**********  Model {i + 1} with loss {loss} **********")
                self.logger.info(f"Parameters {print_dictionary(learner_params)}")
                for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                    train_index = train_index[:train_size]
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = self.base_detector(**learner_params)
                    X_train, y_train = self.reduce_dataset(X_train, y_train)
                    X_test, y_test = self.reduce_dataset(X_test, y_test)
                    model.fit(X=X_train, y=y_train)
                    p_pred, y_pred = get_scores(X_test, model)
                    self.logger.info(f"************************* Split {k + 1} **************************")
                    self.evaluate_scores(X_test, X_train, y_test, y_train, y_pred, p_pred, model, i)
                    if i == 0:
                        self.calculate_random_classifier_accuracy(X_train, y_train, X_test, y_test)
                        self.calculate_majority_voting_accuracy(X_train, y_train, X_test, y_test)
                    directory_path = learner_params.get('base_path', None)
                    if directory_path is not None:
                        try:
                            os.rmdir(directory_path)
                            self.logger.info(f"The directory '{directory_path}' has been removed.")
                        except OSError as e:
                            self.logger.error(f"Error: {directory_path} : {e.strerror}")
            self.store_results()

    def reduce_dataset(self, X, y):
        if X.shape[0] > 4000 and self.base_detector == AutoTabPFNClassifier:
            reduced_size = 4000
            self.logger.info(f"Initial instances {X.shape[0]} reduced to {reduced_size}")
            X, _, y, _ = train_test_split(X, y, train_size=reduced_size,
                                          stratify=y, random_state=self.random_state)
        return X, y
