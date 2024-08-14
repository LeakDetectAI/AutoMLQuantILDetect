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

from autoqild.bayes_search.bayes_search_utils import get_scores, probability_calibration
from autoqild.classifiers import MajorityVoting, RandomClassifier
from autoqild.detectors.utils import *
from autoqild.utilities import *

__all__ = ['InformationLeakageDetector']


class InformationLeakageDetector(metaclass=ABCMeta):
    """
    The `InformationLeakageDetector` class is designed to identify and diagnose information leakage in machine learning models.
    Information leakage occurs when a model inadvertently gains access to information that should not be available during training, leading to overly optimistic performance estimates.

    This class facilitates the detection of such leakage by employing various statistical and machine learning-based methods. It supports multiple detection techniques and is capable of managing the entire process, from cross-validation setup to result storage and evaluation.

    The class is built with flexibility in mind, allowing users to easily extend or customize detection techniques. It includes robust mechanisms for handling result files and backups, ensuring that detection results are safely stored and can be restored if necessary.

    Parameters
    ----------
    padding_name : str
        The name of the padding method used in the experiments to potentially obscure or prevent leakage.

    learner_params : dict
        Parameters related to the machine learning models (learners) used in the leakage detection process.

    fit_params : dict
        Parameters passed to the `fit` method of the models during training.

    hash_value : str
        A unique hash value used to identify and manage result files for a specific experiment.

    cv_iterations : int
        The number of cross-validation iterations to perform during model evaluation.

    n_hypothesis : int
        The number of hypotheses or models to be tested for leakage.

    base_directory : str
        The base directory where result files, logs, and backups are stored.

    detection_method : str
        The method to use for detecting information leakage. Options include:
        - 'paired-t-test': Uses paired t-test to compare the accuracy of models against the majority voting baseline.
        - 'paired-t-test-random': Uses paired t-test to compare the accuracy of models against a random classifier.
        - 'fishers-exact-mean': Applies Fisher's Exact Test on the confusion matrix and computes the mean p-value.
        - 'fishers-exact-median': Applies Fisher's Exact Test on the confusion matrix and computes the median p-value.
        - 'estimated_mutual_information': Estimates mutual information to detect leakage.
        - 'mid_point_mi': Detects leakage using the midpoint mutual information estimation.
        - 'log_loss_mi': Detects leakage using log loss mutual information estimation.
        - 'log_loss_mi_isotonic_regression': Uses log loss mutual information estimation with isotonic regression calibration.
        - 'log_loss_mi_platt_scaling': Uses log loss mutual information estimation with Platt scaling calibration.
        - 'log_loss_mi_beta_calibration': Uses log loss mutual information estimation with beta calibration.
        - 'log_loss_mi_temperature_scaling': Uses log loss mutual information estimation with temperature scaling.
        - 'log_loss_mi_histogram_binning': Uses log loss mutual information estimation with histogram binning.
        - 'p_c_softmax_mi': Uses PC-Softmax mutual information estimation for detection.

    random_state : int or RandomState instance
        Controls the randomness for reproducibility, ensuring consistent results across different runs.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance used for recording the steps and processes of the leakage detection.

    padding_name : str
        The name of the padding method, used for creating unique identifiers and managing results.

    padding_code : str
        A hash code derived from the padding name, used to uniquely identify the experiment.

    fit_params : dict
        Parameters used for fitting the models during training and evaluation.

    learner_params : dict
        Parameters related to the machine learning models (learners) used in the leakage detection process.

    cv_iterations : int
        The number of cross-validation iterations to perform during model evaluation.

    n_hypothesis : int
        The number of hypotheses or models being tested for leakage.

    hash_value : str
        A unique identifier (hash) used to manage and store results.

    random_state : RandomState instance
        Random state instance that ensures reproducibility in cross-validation and other random processes.

    cv_iterator : StratifiedKFold
        Cross-validation iterator that manages the splitting of data into training and test sets.

    estimators : list
        A list of models (estimators) that are evaluated for leakage.

    results : dict
        Dictionary that stores the results of each model's evaluation, organized by metrics.

    base_detector : object
        The underlying model or detector used as the reference for detecting leakage.

    base_directory : str
        The base directory where all results, logs, and backups are stored.

    detection_method : str
        The method used for detecting information leakage, as specified by the user.

    rf_name : str
        The filename where the main results are stored.

    results_file : str
        The full path to the main results file.

    rf_backup_name : str
        The filename where backup results are stored.

    results_file_backup : str
        The full path to the backup results file.

    Private Methods
    ---------------
    __initialize_objects__()
        Initializes the results dictionary for storing metric results.

    __init_results_files__()
        Initializes the results and backup files and restores results from backup if necessary.

    _is_fitted_()
        Checks if the detector has already been fitted by verifying the existence of results files.

    __create_results_from_backup__()
        Creates results files from backup if the main results file is missing or incomplete.

    __update_backup_file__()
        Updates the backup results file with the latest results from the main results file.

    __format_name__(padding_name)
        Formats the padding name and generates a corresponding hash code.

    __close_file__()
        Safely closes the main results file.

    __close_backup_file__()
        Safely closes the backup results file.

    __read_majority_accuracies__()
        Reads and returns the accuracy scores from the majority voting classifier.

    __read_random_accuracies__()
        Reads and returns the accuracy scores from the random classifier.

    __get_training_dataset__(X, y)
        Splits the data into training and test sets using cross-validation.

    __store_results__()
        Stores the evaluation results into the main results file and updates the backup.

    __allkeys__(obj)
        Recursively finds all keys in an h5py.Group object.

    __read_results_file__(detection_method)
        Reads and returns the results for the specified detection method.

    __calculate_majority_voting_accuracy__(X_train, y_train, X_test, y_test)
        Calculates and logs the accuracy of a majority voting classifier.

    __calculate_random_classifier_accuracy__(X_train, y_train, X_test, y_test)
        Calculates and logs the accuracy of a random classifier.
    """
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, detection_method, random_state, **kwargs):

        self.logger = logging.getLogger(InformationLeakageDetector.__name__)
        self.padding_name, self.padding_code = self.__format_name__(padding_name)
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
        self.__create_results_from_backup__()

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
            self.__close_file__()
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
                self.__close_file__()
        self.logger.info(f"++++++++++++++++++++ _is_fitted_ {np.all(conditions_vals)} +++++++++++++++++++++++++++++")

        return np.all(conditions_vals)

    def __create_results_from_backup__(self):
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

    def __update_backup_file__(self):
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
            self.__close_file__()
            self.__close_backup_file__()
        else:
            self.logger.info(f"Result File does not exists {self.rf_name}")

    def __format_name__(self, padding_name):
        padding_name = '_'.join(padding_name.split(' ')).lower()
        padding_name = padding_name.replace(" ", "")
        hash_object = hashlib.sha1()
        hash_object.update(padding_name.encode())
        hex_dig = str(hash_object.hexdigest())[:16]
        # self.logger.info(   "Job_id {} Hash_string {}".format(job.get("job_id", None), str(hex_dig)))
        self.logger.info(f"For padding name {padding_name} the hex value is {hex_dig}")
        return padding_name, hex_dig

    def __close_file__(self):
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

    def __close_backup_file__(self):
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

    def __read_majority_accuracies__(self):
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
            self.__close_file__()
            return accuracies
        else:
            raise ValueError(f"The results are not found at the path {self.rf_name}")

    def __read_random_accuracies__(self):
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
            self.__close_file__()
            return accuracies
        else:
            raise ValueError(f"The results are not found at the path {self.rf_name}")

    def __get_training_dataset__(self, X, y):
        lengths = []
        for i, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
            lengths.append(len(train_index))
        test_size = X.shape[0] - np.min(lengths)
        self.logger.info(f"Test size {test_size} Train sizes {lengths}")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        train_index, test_index = list(sss.split(X, y))[0]
        return X[train_index], y[train_index]

    def __store_results__(self):
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
        self.__close_file__()
        self.__update_backup_file__()

    def __allkeys__(self, obj):
        "Recursively find all keys in an h5py.Group."
        keys = (obj.name,)
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Group):
                    keys = keys + self.__allkeys__(value)
                else:
                    keys = keys + (value.name,)
        return keys

    def __read_results_file__(self, detection_method):
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
            self.__close_file__()
            return model_results
        else:
            raise ValueError(f"The results are not found at the path {self.results_file}")

    def __calculate_majority_voting_accuracy__(self, X_train, y_train, X_test, y_test):
        estimator = MajorityVoting()
        estimator.fit(X_train, y_train)
        p_pred, y_pred = get_scores(X_test, estimator)
        accuracy = accuracy_score(y_test, y_pred)
        self.results[MAJORITY_VOTING][ACCURACY].append(accuracy)
        self.logger.info(f"Majority Voting Performance Metric {ACCURACY}: Value {accuracy}")

    def __calculate_random_classifier_accuracy__(self, X_train, y_train, X_test, y_test):
        estimator = RandomClassifier()
        estimator.fit(X_train, y_train)
        p_pred, y_pred = get_scores(X_test, estimator)
        accuracy = accuracy_score(y_test, y_pred)
        self.results[RANDOM_CLASSIFIER][ACCURACY].append(accuracy)
        self.logger.info(f"Random Classifier Performance Metric {ACCURACY}: Value {accuracy}")

    def perform_hyperparameter_optimization(self, X, y):
        """
        Performs hyperparameter optimization using Bayesian search to identify the best model parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to be used for training during hyperparameter optimization.

        y : array-like, shape (n_samples,)
            The target values (class labels) corresponding to X.

        Returns
        -------
        int
            The size of the training dataset after the reduction (if applicable).
        """
        raise NotImplemented

    def fit(self, X, y):
        """
        Fits the model to the provided data using cross-validation and the specified detection method.
        This method also performs hyperparameter optimization if the model is not already fitted.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to be used for training the models.

        y : array-like, shape (n_samples,)
            The target values (class labels) corresponding to X.
        """
        raise NotImplemented

    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, n_model):
        """
        Evaluates the performance of a model using various metrics and stores the results.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            The input data used for testing.

        X_train : array-like, shape (n_samples, n_features)
            The input data used for training.

        y_test : array-like, shape (n_samples,)
            The true target values for the test data.

        y_train : array-like, shape (n_samples,)
            The true target values for the training data.

        y_pred : array-like, shape (n_samples,)
            The predicted target values for the test data.

        p_pred : array-like, shape (n_samples, n_classes)
            The predicted probabilities for the test data.

        model : object
            The trained model that is being evaluated.

        n_model : int
            The index of the model in the list of evaluated models.
        """

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

    def detect(self):
        """
        Executes the detection process to identify potential information leakage using the specified method.
        This function applies statistical tests to determine if there is significant leakage.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - A boolean indicating whether any models showed significant leakage.
            - An integer representing the number of models that were flagged for leakage.
        """
        # change for including holm-bonnfernoi
        def holm_bonferroni(p_values):
            reject, pvals_corrected, _, alpha = multipletests(p_values, 0.01, method='holm', is_sorted=False)
            reject = [False] * len(p_values) + list(reject)
            pvals_corrected = [1.0] * len(p_values) + list(pvals_corrected)
            return p_values, pvals_corrected, reject

        n_training_folds = self.cv_iterations - 1
        n_test_folds = 1
        model_results = self.__read_results_file__(self.detection_method)
        model_p_values = {}
        for model_name, metric_vals in model_results.items():
            p_value = 1.0
            if self.detection_method in mi_leakage_detection_methods.keys():
                base_mi = self.random_state.rand(len(metric_vals)) * 1e-2
                p_value = paired_ttest(base_mi, metric_vals, n_training_folds, n_test_folds, correction=True)
                self.logger.info("Normal Paired T-Test for MI estimation Technique")
            elif self.detection_method == PAIRED_TTEST:
                accuracies = self.__read_majority_accuracies__()
                self.logger.info(f"Majority Voting Accuracies {accuracies} learner {metric_vals}")
                p_value = paired_ttest(accuracies, metric_vals, n_training_folds, n_test_folds, correction=True)
                self.logger.info("Paired T-Test for accuracy comparison with majority")
            elif self.detection_method == PAIRED_TTEST_RANDOM:
                accuracies = self.__read_random_accuracies__()
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
