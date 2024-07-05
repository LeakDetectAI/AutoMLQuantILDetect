"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  create_final_results.py --schema=<schema> --bucket_id=<bucket_id>
  create_final_results.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                             Show this screen.
  --schema=<schema>                     Schema containing the job information
  --bucket_id=<bucket_id>                     Schema containing the job information
"""
import inspect
import json
import logging
import os

import numpy as np
from docopt import docopt

from experiments.dbconnection import DBConnector
from experiments.utils import *
from pycilt.constants import *
from pycilt.utils import print_dictionary

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXPERIMENTS = 'experiments'

if __name__ == "__main__":
    arguments = docopt(__doc__)
    schema = arguments["--schema"]
    bucket_id = int(arguments["--bucket_id"])
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, f'create_final_results_{schema}_{bucket_id}.log')
    setup_logging(log_path=log_path)
    logger = logging.getLogger('Experiment')
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    if schema == LEAKAGE_DETECTION_NEW:
        n_hypothesis = 11
    elif schema == LEAKAGE_DETECTION:
        n_hypothesis = 6
    elif schema == LEAKAGE_DETECTION_PADDING:
        n_hypothesis = 11

    logger.info(f"Schema analyzed {schema} n_hypothesis {n_hypothesis}")
    db_connector = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema)
    db_connector.init_connection()
    result_table = f"results.{schema}"
    avail_jobs = f"{schema}.avail_jobs"
    final_result_table = f"results.{schema}_final"
    select_job = f"""SELECT * FROM {result_table} JOIN {avail_jobs} ON {result_table}.job_id = {avail_jobs}.job_id 
                     WHERE {result_table}.job_id % 20 = {bucket_id} order by {result_table}.job_id;"""
    db_connector.cursor_db.execute(select_job)
    final_results = db_connector.cursor_db.fetchall()
    db_connector.cursor_db.execute("select to_regclass(%s)", [final_result_table])
    is_table_exist = bool(db_connector.cursor_db.fetchone()[0])
    if not is_table_exist:
        create_table = f"""CREATE TABLE IF NOT EXISTS {final_result_table}
                        (
                            job_id                        integer not null,
                            n_hypothesis_threshold        integer not null,
                            dataset_id                    integer not null,
                            cluster_id                    integer not null,
                            base_detector                 text    not null,
                            detection_method              text    not null,
                            fold_id                       integer not null,
                            imbalance                     double precision,
                            delay                         double precision,
                            f1score                       double precision,
                            accuracy                      double precision,
                            mathewscorrelationcoefficient double precision,
                            balancedaccuracy              double precision,
                            falsepositiverate             double precision,
                            falsenegativerate             double precision,
                            evaluation_time                double precision,
                            hypothesis                    json
                        );"""
        db_connector.cursor_db.execute(create_table)
        admin_allocation = f"""alter table {final_result_table} owner to autoscaadmin;"""
        db_connector.cursor_db.execute(admin_allocation)
        logger.info(f"Table {final_result_table} created successfully")
        column_names_query = f"SELECT * FROM {final_result_table} LIMIT 0;"
        db_connector.cursor_db.execute(column_names_query)
        column_names = [desc[0] for desc in db_connector.cursor_db.description]
        primary_key = f"ALTER TABLE {final_result_table} ADD CONSTRAINT {schema}_final_pkey " \
                      f"PRIMARY KEY(job_id, n_hypothesis_threshold);"
        db_connector.cursor_db.execute(primary_key)
        logger.info("Primary key constraint added successfully")
        done_results = {}
    else:
        logger.info(f"Table {final_result_table} already exists")
        select_jobs = f"SELECT job_id, COUNT(n_hypothesis_threshold) AS threshold_count FROM " \
                      f"{final_result_table} GROUP BY job_id;"
        db_connector.cursor_db.execute(select_jobs)
        done_results = db_connector.cursor_db.fetchall()
        done_results = {row[0]: row[1] for row in done_results}
    db_connector.close_connection()
    db_connector.init_connection()
    done_jobs = []
    readers = {}
    for result in final_results:
        done_hypothesis = done_results.get(result['job_id'], 0)
        if done_hypothesis == n_hypothesis - 1:
            done_jobs.append(result['job_id'])
            logger.info(f"Results job_id {result['job_id']} already exist")
            continue
        dataset_id = result["dataset_params"].get("dataset_id")
        if dataset_id not in readers.keys():
            dataset_name = result["dataset"]
            dataset_params = result["dataset_params"]
            dataset_params['create_datasets'] = False
            dataset_reader = get_dataset_reader(dataset_name, dataset_params)
            readers[dataset_id] = dataset_reader
        else:
            dataset_reader = readers[dataset_id]
        learning_problem = result["learning_problem"]
        result_new = create_results(result)
        hypothesis = dict(result['hypothesis'])
        logger.info("##########################################################################################")
        result_string = print_dictionary(result, sep='\n', n_keys=19)
        logger.info(f"Creating results from {result_string}")
        for threshold in np.arange(1, n_hypothesis):
            if done_hypothesis != 0:
                if check_entry_exists(db_connector, final_result_table, result_new['job_id'], threshold):
                    logger.info(f"Results for threshold {threshold} and job_id {result_new['job_id']} already exist")
                    continue
            y_true, y_pred = [], []
            for label in dataset_reader.label_mapping.keys():
                if label == dataset_reader.correct_class:
                    continue
                ground_truth = int(label in dataset_reader.vulnerable_classes)
                y_true.append(ground_truth)
                rejected_hypothesis = hypothesis[label]
                y_pred.append(int(rejected_hypothesis >= threshold))
            for metric_name, evaluation_metric in lp_metric_dict[learning_problem].items():
                metric_loss = evaluation_metric(y_true, y_pred)
                if np.isnan(metric_loss) or np.isinf(metric_loss):
                    result_new[metric_name] = "Infinity"
                else:
                    if np.around(metric_loss, 4) == 0.0:
                        result_new[metric_name] = f"{metric_loss}"
                    else:
                        result_new[metric_name] = f"{np.around(metric_loss, 4)}"
            result_new['n_hypothesis_threshold'] = threshold
            result_new['hypothesis'] = json.dumps(result['hypothesis'], cls=NpEncoder)
            result_string = print_dictionary(result_new, sep='\t')
            logger.info(f"Results for threshold {threshold} is: {result_string}")
            insert_results_in_table(db_connector, result_new, final_result_table, logger)
        logger.info("##########################################################################################")
        db_connector.connection.commit()
    db_connector.close_connection()
