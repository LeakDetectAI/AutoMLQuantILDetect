"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  cluster_script.py --cindex=<id> --isgpu=<bool> --schema=<schema>
  cluster_script.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                             Show this screen.
  --cindex=<cindex>                     Index given by the cluster to specify which job
                                        is to be executed [default: 0]
  --isgpu=<bool>                        Boolean to show if the gpu is to be used or not
  --schema=<schema>                     Schema containing the job information
"""
import inspect
import logging
import os
import sys
import traceback
from datetime import datetime

import dill
import h5py
import numpy as np
from docopt import docopt
from autoqild import *

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.dbconnection import DBConnector
from experiments.utils import *

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

if __name__ == "__main__":

    ######################## DOCOPT ARGUMENTS: #################################
    arguments = docopt(__doc__)
    cluster_id = int(arguments["--cindex"])
    is_gpu = bool(int(arguments["--isgpu"]))
    schema = arguments["--schema"]
    ###################### POSTGRESQL PARAMETERS ###############################
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=is_gpu, schema=schema)
    os.environ["HIP_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(5000):
        start = datetime.now()
        dbConnector.job_description = None
        if 'CCS_REQID' in os.environ.keys():
            cluster_id = int(os.environ['CCS_REQID'])
        print("************************************************************************************")
        print(f"Getting the job for iteration {i}")
        dbConnector.fetch_job_arguments(cluster_id=cluster_id)
        if dbConnector.job_description is not None:
            try:
                seed = int(dbConnector.job_description["seed"])
                job_id = int(dbConnector.job_description["job_id"])
                fold_id = int(dbConnector.job_description["fold_id"])
                dataset_name = dbConnector.job_description["dataset"]
                n_inner_folds = int(dbConnector.job_description["inner_folds"])
                dataset_params = dbConnector.job_description["dataset_params"]
                learner_name = dbConnector.job_description["learner"]
                fit_params = dbConnector.job_description["fit_params"]
                learner_params_db = dbConnector.job_description["learner_params"]
                duration = dbConnector.job_description["duration"]
                hp_iters = int(dbConnector.job_description["hp_iters"])
                hp_ranges = dbConnector.job_description["hp_ranges"]
                learning_problem = dbConnector.job_description["learning_problem"]
                experiment_schema = dbConnector.job_description["experiment_schema"]
                experiment_table = dbConnector.job_description["experiment_table"]
                validation_loss = dbConnector.job_description["validation_loss"]
                hash_value = dbConnector.job_description["hash_value"]
                LEARNING_PROBLEM = learning_problem.lower()
                if validation_loss == 'None':
                    validation_loss = None
                random_state = np.random.RandomState(seed=seed + fold_id)
                BASE_DIR = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM)
                log_path = os.path.join(BASE_DIR, LOGS_FOLDER, f"{hash_value}.log")
                create_directory_safely(BASE_DIR, False)
                create_directory_safely(log_path, True)

                setup_logging(log_path=log_path)
                setup_random_seed(random_state=random_state)
                logger = logging.getLogger('Experiment')
                print(lp_metric_dict[learning_problem].keys())

                logger.info(f"DB config filePath {config_file_path}")
                logger.info(f"Arguments {arguments}")
                logger.info(f"Job Description {print_dictionary(dbConnector.job_description)}")
                duration = get_duration_seconds(duration)
                dataset_params['random_state'] = random_state
                dataset_params['fold_id'] = fold_id
                # if 'more_instances' in learner_name:
                #    dataset_params["samples_per_class"] = 250 * dataset_params["n_features"]
                dataset_reader = get_dataset_reader(dataset_name, dataset_params)
                X, y = dataset_reader.generate_dataset()
                n_features = X.shape[-1]
                n_classes = len(np.unique(y))
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
                train_index, test_index = list(sss.split(X, y))[0]
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                optimizers_file_path = os.path.join(BASE_DIR, OPTIMIZER_FOLDER, f"{hash_value}.pkl")
                create_directory_safely(optimizers_file_path, True)

                learner = learners[learner_name]
                learner_params = convert_learner_params(learner_params_db)
                learner_params['random_state'] = random_state
                logger.info(f"Time Taken till now: {seconds_to_time(duration_till_now(start))}  seconds")

                if learner in [MultiLayerPerceptron, MineMIEstimator, AutoTabPFNClassifier, MineMIEstimatorHPO]:
                    n_jobs = 1
                else:
                    n_jobs = 10
                logger.info(f"Actual Mutual Information {dataset_reader.get_bayes_mi(MCMC_MI_ESTIMATION)}")
                learner_params = {**learner_params, **dict(n_features=n_features, n_classes=n_classes)}
                if learner in [BayesPredictor, AutoGluonClassifier, MineMIEstimator, MajorityVoting]:
                    if learner == BayesPredictor:
                        learner_params = {'dataset_obj': dataset_reader}
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train)
                        y_true, y_pred, p_pred = estimator.get_bayes_predictor_scores()
                    elif learner == MajorityVoting:
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train)
                        p_pred, y_pred = get_scores(X, estimator)
                        y_true = np.copy(y)
                    elif learner == MineMIEstimator:
                        learner_params = {**learner_params, **dict(n_features=n_features, n_classes=n_classes)}
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train, **fit_params)
                        p_pred, y_pred = get_scores(X, estimator)
                        y_true = np.copy(y)
                    else:
                        estimator = get_automl_learned_estimator(optimizers_file_path, logger)
                        if estimator is None:
                            folder = os.path.join(BASE_DIR, OPTIMIZER_FOLDER, f"{hash_value}gluon")
                            if not os.path.isdir(folder):
                                os.mkdir(folder)
                            logger.info(f"AutoGluon learner params {print_dictionary(learner_params)}")
                            learner_params['output_folder'] = folder
                            learner_params['eval_metric'] = validation_loss
                            estimator = learner(**learner_params)
                            estimator.fit(X_train, y_train, **fit_params)
                            dill.dump(estimator, open(optimizers_file_path, "wb"))
                            logger.info("AutoML pipeline trained and saving the model")
                        else:
                            logger.info("AutoML pipeline trained and reusing it")
                        p_pred, y_pred = get_scores(X, estimator)
                        y_true = np.copy(y)
                else:
                    # inner_cv_iterator = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
                    inner_cv_iterator = StratifiedShuffleSplit(n_splits=n_inner_folds, test_size=0.30,
                                                               random_state=random_state)
                    search_space = create_search_space(hp_ranges, logger)
                    logger.info(f"Search Space {search_space}")

                    estimator = learner(**learner_params)
                    bayes_search_params = dict(estimator=estimator, search_spaces=search_space, n_iter=hp_iters,
                                               scoring=validation_loss, n_jobs=n_jobs, cv=inner_cv_iterator,
                                               error_score=0, random_state=random_state,
                                               optimizers_file_path=optimizers_file_path)
                    bayes_search = BayesSearchCV(**bayes_search_params)
                    search_keys = list(search_space.keys())
                    search_keys.sort()
                    logger.info(f"Search Keys {search_keys}")
                    callback = log_callback(search_keys)
                    try:
                        bayes_search.fit(X_train, y_train, groups=None, callback=callback, **fit_params)
                    except Exception as error:
                        log_exception_error(logger, error)
                        logger.error("Cannot fit the Bayes SearchCV ")
                    logger.info("Fitting the model with best parameters")
                    best_loss, learner_params = update_params_at_k(bayes_search, search_keys, learner_params, k=0)
                    logger.info(f"Setting the best parameters {print_dictionary(learner_params)}")
                    estimator = learner(**learner_params)
                    estimator.fit(X_train, y_train, **fit_params)
                    p_pred, y_pred = get_scores(X, estimator)
                    y_true = np.copy(y)

                if issubclass(learner, MIEstimatorBase):
                    estimated_mi = estimator.estimate_mi(X, y)
                result_file = os.path.join(BASE_DIR, RESULT_FOLDER, f"{hash_value}.h5")
                logger.info(f"Result file {result_file}")

                create_directory_safely(result_file, True)
                f = h5py.File(result_file, 'w')
                f.create_dataset('scores', data=p_pred)
                f.create_dataset('predictions', data=y_pred)
                f.create_dataset('ground_truth', data=y_true)
                f.create_dataset('confusion_matrix', data=confusion_matrix(y_true, y_pred))
                f.close()
                results = {'job_id': str(job_id), 'cluster_id': str(cluster_id)}
                for metric_name, evaluation_metric in lp_metric_dict[learning_problem].items():
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
                                p_pred_cal = probability_calibration(X_train=X_train, y_train=y_train, X_test=X_test,
                                                                     classifier=estimator, calibrator=calibrator)
                                metric_loss = evaluation_metric(y_test, p_pred_cal)
                            except Exception as error:
                                log_exception_error(logger, error)
                                logger.error("Error while calibrating the probabilities setting mi using non"
                                             "calibrated probabilities")
                                metric_loss = evaluation_metric(y_true, p_pred)
                        else:
                            metric_loss = evaluation_metric(y_true, p_pred)
                    elif metric_name in [MCMC_LOG_LOSS, MCMC_MI_ESTIMATION, MCMC_PC_SOFTMAX, MCMC_SOFTMAX]:
                        metric_loss = dataset_reader.get_bayes_mi(metric_name)
                    elif metric_name == ESTIMATED_MUTUAL_INFORMATION:
                        metric_loss = estimated_mi
                    else:
                        if metric_name == F_SCORE:
                            if n_classes > 2:
                                metric_loss = evaluation_metric(y_true, y_pred, average='macro')
                            else:
                                metric_loss = evaluation_metric(y_true, y_pred)
                        else:
                            metric_loss = evaluation_metric(y_true, y_pred)

                    if np.isnan(metric_loss) or np.isinf(metric_loss):
                        results[metric_name] = "\'Infinity\'"
                    else:
                        if np.around(metric_loss, 4) == 0.0:
                            results[metric_name] = f"{metric_loss}"
                        else:
                            results[metric_name] = f"{np.around(metric_loss, 4)}"
                    logger.info(f"Out of sample error {metric_name} : {metric_loss}")
                    print(f"Out of sample error {metric_name} : {metric_loss}")
                evaluation_time = dbConnector.mark_running_job_finished(job_id, start)
                dbConnector.insert_results(experiment_schema=experiment_schema, experiment_table=experiment_table,
                                           results=results)
                logger.info(f"Job finished")
                print(f"Job finished")
            except Exception as e:
                if hasattr(e, 'message'):
                    message = e.message
                else:
                    message = e
                logger.error(traceback.format_exc())
                message = "exception{}".format(str(message))
                dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
            except:
                logger.error(traceback.format_exc())
                message = "exception{}".format(sys.exc_info()[0].__name__)
                dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
            finally:
                if "224" in str(cluster_id):
                    f = open("{}/.hash_value".format(os.environ['HOME']), "w+")
                    f.write(hash_value + "\n")
                    f.close()
