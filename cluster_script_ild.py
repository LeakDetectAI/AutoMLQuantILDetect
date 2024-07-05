"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  cluster_script_ild.py --cindex=<id> --isgpu=<bool> --schema=<schema>
  cluster_script_ild.py (-h | --help)

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
import json
import logging
import os
import pickle as pk
import re
import sys
import traceback
from datetime import datetime

import numpy as np
from docopt import docopt
from pycilt import *

from experiments.dbconnection import DBConnector
from experiments.utils import *
from experiments.utils import leakage_detectors

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LOGS_FOLDER = 'logs'
RESULT_FOLDER = 'results'
EXPERIMENTS = 'experiments'
OPTIMIZER_FOLDER = 'optimizers'

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
    for i in range(400):
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
                dataset_name = dbConnector.job_description["dataset"]
                dataset_params = dbConnector.job_description["dataset_params"]

                base_learner = dbConnector.job_description["base_learner"]
                detection_method = dbConnector.job_description["detection_method"]
                learning_problem = dbConnector.job_description["learning_problem"]

                fit_params = dbConnector.job_description["fit_params"]
                learner_params_db = dbConnector.job_description["learner_params"]
                hp_ranges = dbConnector.job_description["hp_ranges"]
                hp_iters = int(dbConnector.job_description["hp_iters"])
                n_inner_folds = int(dbConnector.job_description["inner_folds"])

                cv_iterations = int(dbConnector.job_description["cv_iterations"])
                n_hypothesis = int(dbConnector.job_description["n_hypothesis"])
                experiment_schema = dbConnector.job_description["experiment_schema"]
                experiment_table = dbConnector.job_description["experiment_table"]
                validation_loss = dbConnector.job_description["validation_loss"]
                hash_value = dbConnector.job_description["hash_value"]
                fold_id = 0
                if "fold_id" in dbConnector.job_description.keys():
                    fold_id = int(dbConnector.job_description["fold_id"])
                LEARNING_PROBLEM = learning_problem.lower()
                if validation_loss == 'None':
                    validation_loss = None
                random_state = np.random.RandomState(seed=(seed + dataset_params.get('dataset_id', 0) + fold_id))
                # Generate different seeds for given random_states
                BASE_DIR = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM, base_learner.lower())
                create_directory_safely(BASE_DIR, False)
                log_path = os.path.join(BASE_DIR, LOGS_FOLDER, f"{hash_value}.log")
                create_directory_safely(log_path, True)
                setup_logging(log_path=log_path)
                time_taken = get_time_taken(log_path)
                other_detection_method_condition = detection_method not in [
                    re.sub(r'(?<!^)(?=[A-Z])', '_', FISHER_EXACT_TEST_MEDIAN).lower(),
                    re.sub(r'(?<!^)(?=[A-Z])', '_', ESTIMATED_MUTUAL_INFORMATION).lower()]
                if other_detection_method_condition:
                    log_path = os.path.join(BASE_DIR, LOGS_FOLDER, f"{hash_value}_{detection_method}.log")
                    setup_logging(log_path=log_path)

                setup_random_seed(random_state=random_state)
                logger = logging.getLogger('Experiment')
                logger.info(f"Time Taken till old: {time_taken} seconds")
                logger.info(f"DB config filePath {config_file_path}")
                logger.info(f"Arguments {arguments}")
                logger.info(f"Job Description {print_dictionary(dbConnector.job_description)}")

                dataset_params['random_state'] = random_state
                dataset_reader = get_dataset_reader(dataset_name, dataset_params)
                search_space = create_search_space(hp_ranges, logger)

                # n_features = X.shape[-1]
                # n_classes = len(np.unique(y))
                ild_learner = leakage_detectors[base_learner]
                learner_params = convert_learner_params(learner_params_db)
                learner_params['random_state'] = random_state
                learner_params = {**learner_params, **dict(n_features=dataset_reader.n_features, n_classes=2)}
                detector_params = {'mi_technique': base_learner, 'learner_params': learner_params,
                                   'fit_params': fit_params, 'hash_value': hash_value, 'cv_iterations': cv_iterations,
                                   'n_hypothesis': n_hypothesis, 'base_directory': BASE_DIR,
                                   'search_space': search_space, 'hp_iters': hp_iters, 'n_inner_folds': n_inner_folds,
                                   'validation_loss': validation_loss, 'detection_method': detection_method}
                detector_params = convert_learner_params(detector_params)
                logger.info(f"Time Taken till now: {seconds_to_time(duration_till_now(start))}  seconds")
                y_true = []
                y_pred = []
                values_of_m = {}
                for label, (X, y) in dataset_reader.dataset_dictionary.items():
                    ground_truth = label in dataset_reader.vulnerable_classes
                    logger.info(f"Running the detector for label {label} vulnerable {ground_truth}")
                    detector_params['padding_name'] = label
                    ild_model = ild_learner(**detector_params)
                    if not ild_model._is_fitted_ and job_id >= 1080:
                        raise NotImplementedError(f"Model not fitted for the padding_name {label} "
                                                  f"need to retrain them")
                    ild_model.fit(X, y)
                    predicted_decision, n_hypothesis_detection = ild_model.detect()
                    logger.info(f"The label is vulnerable {ground_truth} and predicted {predicted_decision}")
                    y_true.append(ground_truth)
                    y_pred.append(predicted_decision)
                    values_of_m[label] = n_hypothesis_detection

                result_file = os.path.join(BASE_DIR, RESULT_FOLDER, f"{hash_value}.pkl")
                logger.info(f"Result file {result_file}")
                create_directory_safely(result_file, True)
                results = {'y_pred': y_pred, 'y_true': y_true, 'values_of_m': values_of_m}
                logger.info(f"Results {print_dictionary(results)}")
                file = open(result_file, 'wb')
                pk.dump(results, file=file)
                file.close()

                results = {'job_id': str(job_id), 'cluster_id': str(cluster_id)}
                results['hypothesis'] = json.dumps(values_of_m, cls=NpEncoder)
                if dataset_name == OPENML_DATASET:
                    results['delay'] = f"{dataset_reader.delay}"
                    results['fold_id'] = f"{dataset_reader.fold_id}"
                if dataset_name == OPENML_PADDING_DATASET:
                    results['server'] = f"{dataset_reader.server}"
                    results['fold_id'] = f"{fold_id}"
                results['imbalance'] = f"{dataset_reader.imbalance}"
                results['base_detector'] = f"{base_learner}"
                results['detection_method'] = f"{detection_method}"
                results['dataset_id'] = f"{dataset_reader.dataset_id}"
                for metric_name, evaluation_metric in lp_metric_dict[learning_problem].items():
                    metric_loss = evaluation_metric(y_true, y_pred)
                    if np.isnan(metric_loss) or np.isinf(metric_loss):
                        results[metric_name] = "Infinity"
                    else:
                        if np.around(metric_loss, 4) == 0.0:
                            results[metric_name] = f"{metric_loss}"
                        else:
                            results[metric_name] = f"{np.around(metric_loss, 4)}"
                    logger.info(f"Out of sample error {metric_name} : {metric_loss}")
                    print(f"Out of sample error {metric_name} : {metric_loss}")

                evaluation_time = dbConnector.mark_running_job_finished(job_id=job_id, start=start,
                                                                        old_time_take=time_taken)
                results['evaluation_time'] = f"{evaluation_time}"
                dbConnector.insert_results(experiment_schema=experiment_schema, experiment_table=experiment_table,
                                           results=results)
                logger.info("Job finished")
                print(f"Job finished")
            except Exception as e:
                if hasattr(e, 'message'):
                    message = e.message
                else:
                    message = e
                logger.error(traceback.format_exc())
                lowest_job_id, status = dbConnector.get_lowest_job_id_with_hash(hash_value=hash_value)
                message = f"exception\tlowest_job_id_{lowest_job_id} \n with status {status}\t{str(message)}"
                if isinstance(e, NotImplementedError):
                    dbConnector.append_error_string_in_running_job2(job_id=job_id, error_message=message)
                    if "running" not in status.lower():
                        dbConnector.append_error_string_in_running_job(job_id=lowest_job_id, error_message=message)
                else:
                    dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
            except:
                logger.error(traceback.format_exc())
                message = f"exception{sys.exc_info()[0].__name__}"
                dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
            finally:
                if "224" in str(cluster_id):
                    f = open("{}/.hash_value".format(os.environ['HOME']), "w+")
                    f.write(hash_value + "\n")
                    f.close()
