import fcntl
import hashlib
import logging
import os
import shutil
from abc import ABCMeta

import h5py
import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests

from autoqild.detectors.utils import *
from autoqild.bayes_search.bayes_search_utils import get_scores, probability_calibration
from autoqild.classifiers import MajorityVoting, RandomClassifier
from autoqild.utilities import *

__all__ = ['InformationLeakageDetector']


class InformationLeakageDetector(metaclass=ABCMeta):
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, detection_method, random_state, **kwargs):

        self.logger = logging.getLogger(InformationLeakageDetector.__name__)
        self.padding_name, self.padding_code = self.format_name(padding_name)
        self.fit_params = fit_params
        self.learner_params = learner_params
        self.cv_iterations = cv_iterations
        self.n_hypothesis = n_hypothesis

        self.hash_value = hash_value
        self.random_state = check_random_state(random_state)
        self.cv_iterator = StratifiedKFold(n_splits=self.cv_iterations, random_state=random_state, shuffle=True)

        self.estimators = []
        self.results = {}
        self.base_detector = None
        self.base_directory = base_directory
        self.detection_method = detection_method
        self.__init_results_files__()
        self.__initialize_objects__()

    def __initialize_objects__(self):
        for i in range(self.n_hypothesis):
            self.results[f'model_{i}'] = {}
            for metric_name, evaluation_metric in mi_estimation_metrics.items():
                self.results[f'model_{i}'][metric_name] = []
        self.results[MAJORITY_VOTING] = {}
        self.results[MAJORITY_VOTING][ACCURACY] = []
        self.results[RANDOM_CLASSIFIER] = {}
        self.results[RANDOM_CLASSIFIER][ACCURACY] = []

    def __init_results_files__(self):
        if self.detection_method not in leakage_detection_methods.keys():
            raise ValueError(f"Invalid Detection Method {self.detection_method}")
        hv_dm = leakage_detection_names[self.detection_method]
        self.rf_name = f"{self.hash_value}_eval.h5"
        self.results_file = os.path.join(self.base_directory, RESULT_FOLDER, hv_dm, self.rf_name)
        self.rf_backup_name = f"{self.hash_value}_backup.h5"
        self.results_file_backup = os.path.join(self.base_directory, RESULT_FOLDER, self.rf_backup_name)
        create_directory_safely(self.results_file, True)
        create_directory_safely(self.results_file_backup, True)
        self.create_results_from_backup()

    @property
    def _is_fitted_(self) -> bool:
        self.logger.info(f"++++++++++++++++++++++++++++++++ _is_fitted_ function ++++++++++++++++++++++++++++++++")

        self.logger.info(f"Checking main file {self.rf_name} for results for padding {self.padding_name}")
        check_and_delete_corrupt_h5_file(self.results_file, self.logger)
        conditions = {"os.path.exists(self.results_file)": os.path.exists(self.results_file)}
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            conditions[f"{self.padding_code} in file"] = self.padding_code in file
            if self.padding_code in file:
                self.logger.info(f"Simulations done for padding label {self.padding_code}")
                for model_name, metric_results in self.results.items():
                    padding_name_group = file[self.padding_code]
                    self.logger.info(f"Check if {model_name} exists in results {model_name in padding_name_group}")
                    conditions[f"{model_name} in {padding_name_group}"] = model_name in padding_name_group
                    if model_name in padding_name_group:
                        model_group = padding_name_group.get(model_name)
                        self.logger.info(f"Predictions done for model {model_name}")
                        for metric_name, results in metric_results.items():
                            conditions[f"{metric_name} in {model_group}"] = metric_name in model_group
                            self.logger.info(f"Results exists for metric {metric_name}: {metric_name in model_group}")
                            vals = np.array(model_group[metric_name]) #np.array(model_group.get(metric_name))
                            self.logger.info(f"Results {vals} stored for {self.cv_iterations} exist for {len(vals)}")
                            conditions[f"{padding_name_group}_{model_name}_{metric_name} len(vals) == self.cv_iterations"] = len(vals) == self.cv_iterations
            file.close()
            self.close_file()
        conditions_vals = list(conditions.values())
        self.logger.info(f"Results for padding {self.padding_name} {not np.all(conditions_vals)} "
                         f"Coniditions: {print_dictionary(conditions)}")
        if os.path.exists(self.results_file) and not np.all(conditions_vals):
            if os.path.exists(self.results_file):
                file = h5py.File(self.results_file, 'w')
                if self.padding_code in file:
                    del file[self.padding_code]
                    self.logger.info(f"Results for padding {self.padding_name} removed since it is incomplete "
                                     f"{not np.all(conditions_vals)} {print_dictionary(conditions)}")
                file.close()
                self.close_file()
        self.logger.info(f"++++++++++++++++++++ _is_fitted_ {np.all(conditions_vals)} +++++++++++++++++++++++++++++")

        return np.all(conditions_vals)

    def create_results_from_backup(self):
        check_and_delete_corrupt_h5_file(self.results_file_backup, self.logger)
        dir_path = os.path.dirname(os.path.realpath(self.results_file_backup))

        if os.path.exists(self.results_file_backup):
            if not self._is_fitted_:
                source = h5py.File(self.results_file_backup, 'r')
                # Apply a shared lock on the source file
                fcntl.flock(source.id.get_vfd_handle(), fcntl.LOCK_SH)
                self.logger.info(f"Locked the backup file: {self.rf_backup_name}")
                # Perform the file copy operation
                shutil.copy(self.results_file_backup, self.results_file)
                self.logger.info(f"Copied the file from {self.rf_backup_name} to {dir_path}, {self.rf_name}")
                # Release the lock on the source file
                fcntl.flock(source.id.get_vfd_handle(), fcntl.LOCK_UN)
                source.close()
                self.logger.info(f"Unlocked the backup file: {self.rf_backup_name}")
            else:
                self.logger.info(f"Latest results already complete for the {self.padding_name}")
        else:
            self.logger.info(f"Backup results file does not exists {self.rf_backup_name}")

    def update_backup_file(self):
        if os.path.exists(self.results_file):
            if os.path.isfile(self.results_file_backup):
                dest_h5 = h5py.File(self.results_file_backup, 'a')
                if self.padding_code in dest_h5:
                    del dest_h5[self.padding_code]
                    self.logger.info(f"Deleting old results for the padding from backup file {self.padding_name}")
                dest_h5.close()
            source_h5 = h5py.File(self.results_file, 'r')
            destination_h5 = h5py.File(self.results_file_backup, 'a')
            if self.padding_code in source_h5:
                source_group = source_h5[self.padding_code]
                destination_h5.copy(source_group, destination_h5, name=self.padding_code)
                self.logger.info(f"Group '{self.padding_code}' copied to backup result file "
                                 f"{self.rf_backup_name} for padding {self.padding_name}")
            else:
                self.logger.info(f"Group '{self.padding_code}' does not exist in the results file "
                                 f"{self.rf_name} for padding {self.padding_name}")
            # Close the HDF5 files
            source_h5.close()
            destination_h5.close()
            self.close_file()
            self.close_back_file()
        else:
            self.logger.info(f"Result File does not exists {self.rf_name}")

    def format_name(self, padding_name):
        padding_name = '_'.join(padding_name.split(' ')).lower()
        padding_name = padding_name.replace(" ", "")
        hash_object = hashlib.sha1()
        hash_object.update(padding_name.encode())
        hex_dig = str(hash_object.hexdigest())[:16]
        # self.logger.info(   "Job_id {} Hash_string {}".format(job.get("job_id", None), str(hex_dig)))
        self.logger.info(f"For padding name {padding_name} the hex value is {hex_dig}")
        return padding_name, hex_dig

    def get_training_dataset(self, X, y):
        lengths = []
        for i, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
            lengths.append(len(train_index))
        test_size = X.shape[0] - np.min(lengths)
        self.logger.info(f"Test size {test_size} Train sizes {lengths}")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        train_index, test_index = list(sss.split(X, y))[0]
        return X[train_index], y[train_index]

    def calculate_majority_voting_accuracy(self, X_train, y_train, X_test, y_test):
        estimator = MajorityVoting()
        estimator.fit(X_train, y_train)
        p_pred, y_pred = get_scores(X_test, estimator)
        accuracy = accuracy_score(y_test, y_pred)
        self.results[MAJORITY_VOTING][ACCURACY].append(accuracy)
        self.logger.info(f"Majority Voting Performance Metric {ACCURACY}: Value {accuracy}")

    def calculate_random_classifier_accuracy(self, X_train, y_train, X_test, y_test):
        estimator = RandomClassifier()
        estimator.fit(X_train, y_train)
        p_pred, y_pred = get_scores(X_test, estimator)
        accuracy = accuracy_score(y_test, y_pred)
        self.results[RANDOM_CLASSIFIER][ACCURACY].append(accuracy)
        self.logger.info(f"Random Classifier Performance Metric {ACCURACY}: Value {accuracy}")

    def perform_hyperparameter_optimization(self, X, y):
        raise NotImplemented

    def fit(self, X, y):
        raise NotImplemented

    def close_file(self):
        try:
            file = h5py.File(self.results_file, 'r')
            is_open = file.id.valid
            if is_open:
                self.logger.info("The result file is open closing it")
                file.close()
            else:
                self.logger.info("The result file is not open.")
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Cannot open the result file since it does not exist")

    def close_back_file(self):
        try:
            file = h5py.File(self.results_file_backup, 'r')
            is_open = file.id.valid
            if is_open:
                self.logger.info("The backup file is open closing it")
                file.close()
            else:
                self.logger.info("The backup file is not open.")
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Cannot open the backup file since it does not exist")

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        model_name = list(self.results.keys())[n_model]
        self.logger.info(f"Appending results for model {model_name}")
        for metric_name, evaluation_metric in mi_estimation_metrics.items():
            if LOG_LOSS_MI_ESTIMATION in metric_name or PC_SOFTMAX_MI_ESTIMATION in metric_name:
                calibrator_technique = None
                for key in calibrators.keys():
                    if key in metric_name:
                        calibrator_technique = key
                if calibrator_technique is not None:
                    calibrator = calibrators[calibrator_technique]
                    c_params = calibrator_params[calibrator_technique]
                    calibrator = calibrator(**c_params)
                    try:
                        p_pred_cal = probability_calibration(X_train=X_train, y_train=y_train,
                                                             X_test=X_test, classifier=model,
                                                             calibrator=calibrator)
                        metric_loss = evaluation_metric(y_test, p_pred_cal)
                    except Exception as error:
                        log_exception_error(self.logger, error)
                        self.logger.error("Error while calibrating the probabilities")
                        metric_loss = evaluation_metric(y_test, p_pred)
                else:
                    metric_loss = evaluation_metric(y_test, p_pred)
            else:
                metric_loss = evaluation_metric(y_test, y_pred)
            if metric_name == CONFUSION_MATRIX:
                # metric_loss = np.array(metric_loss)
                (tn, fp, fn, tp) = metric_loss.ravel()
                cm_string = f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}"
                metric_loss = [tn, fp, fn, tp]
                self.logger.info(f"Metric {metric_name}: Value: {cm_string}")
            else:
                self.logger.info(f"Metric {metric_name}: Value: {metric_loss}")
            self.results[model_name][metric_name].append(metric_loss)

    def store_results(self):
        self.logger.info(f"Result file {self.rf_name}")
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r+')
        else:
            file = h5py.File(self.results_file, 'w')
        try:
            self.logger.info(f"{self.padding_code} in {file}: {self.padding_code in file}")
            if self.padding_code not in file:
                padding_name_group = file.create_group(self.padding_code)
            else:
                padding_name_group = file.get(self.padding_code)
            for model_name, metric_results in self.results.items():
                self.logger.info(f"{model_name} in {padding_name_group}: {model_name in padding_name_group}")
                if model_name not in padding_name_group:
                    model_group = padding_name_group.create_group(model_name)
                    self.logger.info(f"Creating model group {model_name} results {model_group}")
                else:
                    model_group = padding_name_group.get(model_name)
                    self.logger.info(f"Extracting model group {model_name} results {model_group}")
                for metric_name, results in metric_results.items():
                    self.logger.info(f"Storing results {metric_name} results {np.array(results)}")
                    if metric_name in model_group:
                        del model_group[metric_name]
                    model_group.create_dataset(metric_name, data=np.array(results))
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Problem creating the dataset ")
        finally:
            file.close()
        self.close_file()
        self.update_backup_file()

    def allkeys(self, obj):
        "Recursively find all keys in an h5py.Group."
        keys = (obj.name,)
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Group):
                    keys = keys + self.allkeys(value)
                else:
                    keys = keys + (value.name,)
        return keys

    def read_results_file(self, detection_method):
        metric_name = leakage_detection_methods[detection_method]
        self.logger.info(f"For the detection method {detection_method}, metric {metric_name}")
        model_results = {}
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            # self.logger.error(self.allkeys(file))
            padding_name_group = file[self.padding_code]
            # self.logger.error(self.allkeys(padding_name_group))
            for model_name in self.results.keys():
                if model_name in [MAJORITY_VOTING, RANDOM_CLASSIFIER]:
                    continue
                model_group = padding_name_group[model_name]
                # self.logger.error(self.allkeys(model_group))
                try:
                    model_results[model_name] = np.array(model_group[metric_name])
                except KeyError as e:
                    log_exception_error(self.logger, e)
                    self.logger.error(f"Error while getting the metric {metric_name} for the"
                                      f"detection method {detection_method}")
                    raise ValueError(f"Provided Metric Name {metric_name} is not applicable "
                                     f"for current base detector {self.base_detector} "
                                     f"so cannot apply the provided detection method {detection_method}")
            file.close()
            self.close_file()
            return model_results
        else:
            raise ValueError(f"The results are not found at the path {self.results_file}")

    def read_majority_accuracies(self):
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            # self.logger.error(self.allkeys(file))
            padding_name_group = file[self.padding_code]
            # self.logger.error(self.allkeys(padding_name_group))
            try:
                model_group = padding_name_group[MAJORITY_VOTING]
                accuracies = np.array(model_group[ACCURACY])

            except KeyError as e:
                log_exception_error(self.logger, e)
                self.logger.error(f"Error while getting the metric {ACCURACY} for the"
                                  f"detection method {PAIRED_TTEST}")
            file.close()
            self.close_file()
            return accuracies
        else:
            raise ValueError(f"The results are not found at the path {self.rf_name}")

    def read_random_accuracies(self):
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            # self.logger.error(self.allkeys(file))
            padding_name_group = file[self.padding_code]
            # self.logger.error(self.allkeys(padding_name_group))
            try:
                model_group = padding_name_group[RANDOM_CLASSIFIER]
                accuracies = np.array(model_group[ACCURACY])
            except KeyError as e:
                log_exception_error(self.logger, e)
                self.logger.error(f"Error while getting the metric {ACCURACY} for the"
                                  f"detection method {PAIRED_TTEST}")
            file.close()
            self.close_file()
            return accuracies
        else:
            raise ValueError(f"The results are not found at the path {self.rf_name}")

    def detect(self):
        # change for including holm-bonnfernoi
        def holm_bonferroni(p_values):
            reject, pvals_corrected, _, alpha = multipletests(p_values, 0.01, method='holm', is_sorted=False)
            reject = [False] * len(p_values) + list(reject)
            pvals_corrected = [1.0] * len(p_values) + list(pvals_corrected)
            return p_values, pvals_corrected, reject

        n_training_folds = self.cv_iterations - 1
        n_test_folds = 1
        model_results = self.read_results_file(self.detection_method)
        model_p_values = {}
        for model_name, metric_vals in model_results.items():
            p_value = 1.0
            if self.detection_method in mi_leakage_detection_methods.keys():
                base_mi = self.random_state.rand(len(metric_vals)) * 1e-2
                p_value = paired_ttest(base_mi, metric_vals, n_training_folds, n_test_folds, correction=True)
                self.logger.info("Normal Paired T-Test for MI estimation Technique")
            elif self.detection_method == PAIRED_TTEST:
                accuracies = self.read_majority_accuracies()
                self.logger.info(f"Majority Voting Accuracies {accuracies} learner {metric_vals}")
                p_value = paired_ttest(accuracies, metric_vals, n_training_folds, n_test_folds, correction=True)
                self.logger.info("Paired T-Test for accuracy comparison with majority")
            elif self.detection_method == PAIRED_TTEST_RANDOM:
                accuracies = self.read_random_accuracies()
                self.logger.info(f"Random Classifier Accuracies {accuracies} learner {metric_vals}")
                p_value = paired_ttest(accuracies, metric_vals, n_training_folds, n_test_folds, correction=True)
                self.logger.info("Paired T-Test for accuracy comparison with random")
            elif self.detection_method in [FISHER_EXACT_TEST_MEAN, FISHER_EXACT_TEST_MEDIAN]:
                self.logger.info("Fisher's Exact-Test for confusion matrix")
                metric_vals = [np.array([[tn, fp], [fn, tp]]) for [tn, fp, fn, tp] in metric_vals]
                p_values = np.array([fisher_exact(cm)[1] for cm in metric_vals])
                if self.detection_method == FISHER_EXACT_TEST_MEAN:
                    p_value = np.mean(p_values)
                elif self.detection_method == FISHER_EXACT_TEST_MEDIAN:
                    p_value = np.median(p_values)
            model_p_values[model_name] = p_value
            self.logger.info(f"Model {model_name} p-value {p_value}")
        p_vals, pvals_corrected, rejected = holm_bonferroni(list(model_p_values.values()))
        return np.any(rejected), np.sum(rejected)
