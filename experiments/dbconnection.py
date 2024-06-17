import hashlib
import json
import logging
import os
import re
from abc import ABCMeta
from datetime import timedelta, datetime

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from pycilt.constants import *
from pycilt.dataset_readers import GEN_TYPES, generate_samples_per_class
from pycilt.detectors.utils import leakage_detection_methods
from pycilt.utils import print_dictionary

from experiments.utils import get_duration_seconds, duration_till_now, get_openml_datasets, NpEncoder, \
    get_openml_padding_datasets


class DBConnector(metaclass=ABCMeta):
    def __init__(self, config_file_path, is_gpu=False, schema="master", create_hash_list=False, **kwargs):
        self.logger = logging.getLogger("DBConnector")
        self.is_gpu = is_gpu
        self.schema = schema
        self.job_description = None
        self.connection = None
        self.cursor_db = None
        self.create_hash_list = create_hash_list
        if os.path.isfile(config_file_path):
            config_file = open(config_file_path, "r")
            config = config_file.read().replace("\n", "")
            self.logger.info("Config {}".format(config))
            self.connect_params = json.loads(config)
            self.logger.info("Connection Successful")
        else:
            raise ValueError(
                "File does not exist for the configuration of the database"
            )
        if create_hash_list:
            self.current_hash_values = self.create_current_job_list()
        else:
            self.current_hash_values = []

    def create_current_job_list(self):
        avail_jobs = "{}.avail_jobs".format(self.schema)
        self.init_connection()
        select_job = f"SELECT * FROM {avail_jobs} ORDER  BY {avail_jobs}.job_id"
        self.cursor_db.execute(select_job)
        jobs_check = self.cursor_db.fetchall()
        self.close_connection()
        current_hash_values = []
        for job_c in jobs_check:
            job_c = dict(job_c)
            if self.schema in [LEAKAGE_DETECTION, LEAKAGE_DETECTION_NEW, LEAKAGE_DETECTION_PADDING]:
                hash_value = self.get_hash_value_for_job_ild_check(job_c)
            else:
                hash_value = self.get_hash_value_for_job(job_c)
            current_hash_values.append(hash_value)
        return current_hash_values

    def get_hash_value_for_job(self, job):
        keys = [
            "fold_id",
            "base_learner",
            "learner",
            "dataset_params",
            "fit_params",
            "learner_params",
            "hp_ranges",
            "inner_folds",
            "validation_loss",
            "dataset"
        ]
        hash_string = ""
        for k in keys:
            if k in job.keys():
                hash_string = hash_string + str(k) + ":" + str(job[k])
        hash_object = hashlib.sha1(hash_string.encode())
        hex_dig = hash_object.hexdigest()
        # self.logger.info(   "Job_id {} Hash_string {}".format(job.get("job_id", None), str(hex_dig)))
        return str(hex_dig)

    def get_hash_value_for_job_ild_check(self, job):
        keys = [
            "fold_id",
            "base_learner",
            "learner",
            "dataset_params",
            "fit_params",
            "learner_params",
            "hp_ranges",
            "inner_folds",
            "validation_loss",
            "dataset",
            "detection_method"
        ]
        hash_string = ""
        for k in keys:
            if k in job.keys():
                hash_string = hash_string + str(k) + ":" + str(job[k])
        hash_object = hashlib.sha1(hash_string.encode())
        hex_dig = hash_object.hexdigest()
        # self.logger.info(   "Job_id {} Hash_string {}".format(job.get("job_id", None), str(hex_dig)))
        return str(hex_dig)

    def init_connection(self, cursor_factory=DictCursor):
        self.connection = psycopg2.connect(**self.connect_params)
        if cursor_factory is None:
            self.cursor_db = self.connection.cursor()
        else:
            self.cursor_db = self.connection.cursor(cursor_factory=cursor_factory)

    def close_connection(self):
        self.connection.commit()
        self.cursor_db.close()
        self.connection.close()

    def add_jobs_in_avail_which_failed(self):
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format(self.schema)
        running_jobs = "{}.running_jobs".format(self.schema)
        select_job = f"""SELECT * FROM {avail_jobs} row WHERE EXISTS(SELECT job_id FROM {running_jobs} r 
                         WHERE r.interrupted = FALSE AND r.finished = FALSE AND r.job_id = row.job_id)"""
        self.cursor_db.execute(select_job)
        all_jobs = self.cursor_db.fetchall()
        # print(f"Running jobs are {all_jobs}")
        self.close_connection()
        for job in all_jobs:
            date_time = job["job_allocated_time"]
            duration = get_duration_seconds(job["duration"])
            new_date = date_time + timedelta(seconds=duration)
            if new_date < datetime.now():
                job_id = int(job["job_id"])
                print(
                    "Duration for the Job {} expired so marking it as failed".format(
                        job_id
                    )
                )
                error_message = "exception{}".format("InterruptedDueToSomeError")
                self.append_error_string_in_running_job(job_id=job_id, error_message=error_message)

    def get_job_for_id(self, cluster_id, job_id):
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format(self.schema)
        running_jobs = "{}.running_jobs".format(self.schema)
        select_job = f"""SELECT * FROM {avail_jobs}  WHERE {avail_jobs}.job_id={job_id}"""
        self.cursor_db.execute(select_job)

        if self.cursor_db.rowcount == 1:
            try:
                self.job_description = self.cursor_db.fetchall()[0]
                print("Jobs found {}".format(print_dictionary(self.job_description)))
                start = datetime.now()
                update_job = f"""UPDATE {avail_jobs} set job_allocated_time = %s WHERE job_id = %s"""
                self.cursor_db.execute(update_job, (start, job_id))
                select_job = """SELECT * FROM {0} WHERE {0}.job_id = {1} AND {0}.interrupted = {2} AND
                                {0}.finished = {3} FOR UPDATE""".format(running_jobs, job_id, False, True)
                self.cursor_db.execute(select_job)
                running_job = self.cursor_db.fetchall()
                if len(running_job) == 0:
                    self.job_description = None
                    print("The job is not evaluated yet")
                else:
                    print(f"Job with job_id {job_id} present in the updating and row locked")
                    update_job = f"""UPDATE {avail_jobs} set cluster_id = %s, interrupted = %s, finished = %s 
                                    WHERE job_id = %s"""
                    self.cursor_db.execute(update_job, (cluster_id, "FALSE", "FALSE", job_id))
                    if self.cursor_db.rowcount == 1:
                        print(f"The job {job_id} is updated")
                self.close_connection()
            except psycopg2.IntegrityError as e:
                print(f"IntegrityError for the job {job_id}, already assigned to another node error {str(e)}")
                self.job_description = None
                self.connection.rollback()
            except (ValueError, IndexError) as e:
                print(f"Error as the all jobs are already assigned to another nodes {str(e)}")

    def fetch_job_arguments(self, cluster_id):
        # self.add_jobs_in_avail_which_failed()
        self.init_connection()
        avail_jobs = f"{self.schema}.avail_jobs"
        running_jobs = f"{self.schema}.running_jobs"

        select_job = f"""SELECT job_id FROM {avail_jobs} row WHERE (is_gpu = {self.is_gpu}) AND NOT 
                         EXISTS(SELECT job_id FROM {running_jobs} r WHERE r.interrupted = FALSE 
                         AND r.job_id = row.job_id)"""
        print(select_job)
        self.cursor_db.execute(select_job)
        job_ids = [j for i in self.cursor_db.fetchall() for j in i]
        job_ids.sort()
        print(f"jobs available {np.array(job_ids)[:10]}")
        while self.job_description is None:
            try:
                job_id = job_ids[0]
                print(f"Job selected : {job_id}")
                select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.job_id = {job_id}"
                self.cursor_db.execute(select_job)
                self.job_description = self.cursor_db.fetchone()
                # if turn_filter_on:
                #    learner = self.job_description['learner']
                #   if self.schema == MUTUAL_INFORMATION_NEW and learner in LEARNERS:
                #       self.job_description = None
                #        del job_ids[0]
                #        continue
                print(print_dictionary(self.job_description))
                hash_value = self.get_hash_value_for_job(self.job_description)
                self.job_description["hash_value"] = hash_value

                start = datetime.now()
                update_job = f"""UPDATE {avail_jobs} set hash_value = %s, job_allocated_time = %s WHERE job_id = %s"""
                self.cursor_db.execute(update_job, (hash_value, start, job_id))
                select_job = f"""SELECT * FROM {running_jobs} WHERE {running_jobs}.job_id = {job_id} AND 
                                 {running_jobs}.interrupted = {True} FOR UPDATE"""
                self.cursor_db.execute(select_job)
                count_ = len(self.cursor_db.fetchall())
                if count_ == 0:
                    insert_job = f"""INSERT INTO {running_jobs} (job_id, cluster_id ,finished, interrupted) 
                                     VALUES ({job_id}, {cluster_id}, FALSE, FALSE)"""
                    self.cursor_db.execute(insert_job)
                    if self.cursor_db.rowcount == 1:
                        print(f"The job {job_id} is inserted")
                else:
                    print(f"Job with job_id {job_id} present in the updating and row locked")
                    update_job = f"""UPDATE {running_jobs} set cluster_id = %s, interrupted = %s WHERE job_id = %s""".format(

                    )
                    self.cursor_db.execute(update_job, (cluster_id, "FALSE", job_id))
                    if self.cursor_db.rowcount == 1:
                        print(f"The job {job_id} is updated")

                self.close_connection()
            except psycopg2.IntegrityError as e:
                print(f"IntegrityError for the job {job_id}, already assigned to another node error {str(e)}")
                self.job_description = None
                job_ids.remove(job_id)
                self.connection.rollback()
            except (ValueError, IndexError) as e:
                print(f"Error as the all jobs are already assigned to another nodes {str(e)}")
                break

    def mark_running_job_finished(self, job_id, start, old_time_take=0, **kwargs):
        self.init_connection()
        running_jobs = "{}.running_jobs".format(self.schema)
        avail_jobs = "{}.avail_jobs".format(self.schema)
        update_job = f"""UPDATE {running_jobs} set finished = TRUE, interrupted = FALSE  WHERE job_id = {job_id}"""
        self.cursor_db.execute(update_job)
        if self.cursor_db.rowcount == 1:
            self.logger.info(f"The job {job_id} is finished")

        end_time = datetime.now()
        select_job = f"""SELECT evaluation_time from {avail_jobs} WHERE job_id = {job_id}"""
        self.cursor_db.execute(select_job)
        evaluation_time = float(self.cursor_db.fetchone()[0])  # Retrieve the first column value from the result
        old_time_take = np.max([evaluation_time, old_time_take])
        time_taken = duration_till_now(start) + old_time_take
        self.logger.info(f"The job {job_id} is old time taken {old_time_take} time taken {duration_till_now(start)}")
        update_job = f"""UPDATE {avail_jobs} set job_end_time = %s, evaluation_time = %s WHERE 
                         job_id = %s RETURNING evaluation_time"""
        self.cursor_db.execute(update_job, (end_time, time_taken, job_id))
        if self.cursor_db.rowcount == 1:
            evaluation_time = self.cursor_db.fetchone()[0]
            self.logger.info(f"The job {job_id} end time {end_time} is updated, total-time {evaluation_time}")
        self.close_connection()
        return evaluation_time

    def insert_results(self, experiment_schema, experiment_table, results, **kwargs):
        self.init_connection(cursor_factory=None)
        results_table = f"{experiment_schema}.{experiment_table}"
        try:
            keys = list(results.keys())
            values = list(results.values())
            columns = ", ".join(list(results.keys()))
            values_str = []
            for i, (key, val) in enumerate(zip(keys, values)):
                if isinstance(val, dict):
                    val = json.dumps(val, cls=NpEncoder)
                else:
                    val = str(val)
                values_str.append(val)
                if i == 0:
                    str_values = "%s"
                else:
                    str_values = str_values + ", %s"
            insert_result = f"INSERT INTO {results_table} ({columns}) VALUES ({str_values}) RETURNING job_id"
            self.logger.info(f"Inserting results: {insert_result} values {values_str}")
            self.cursor_db.execute(insert_result, tuple(values_str))
            if self.cursor_db.rowcount == 1:
                self.logger.info(f"Results inserted for the job {results['job_id']}")
        except psycopg2.IntegrityError as e:
            self.logger.info(print_dictionary(results))
            self.logger.info(
                f"IntegrityError for the job {results['job_id']}, results already inserted to another node error {str(e)}")
            self.connection.rollback()
            update_str = ""
            values_tuples = []
            for i, col in enumerate(results.keys()):
                if col != "job_id":
                    if (i + 1) == len(results):
                        update_str = update_str + col + " = %s "
                    else:
                        update_str = update_str + col + " = %s, "
                    if "Infinity" in results[col]:
                        results[col] = "Infinity"
                    values_tuples.append(results[col])
            job_id = results["job_id"]
            select_job = f"SELECT * from {results_table} WHERE job_id={job_id}"
            self.cursor_db.execute(select_job)
            old_results = self.cursor_db.fetchone()

            update_result = f"UPDATE {results_table} set {update_str} where job_id= %s "
            self.logger.info(update_result)
            values_tuples.append(results["job_id"])
            self.logger.info(f"Values {tuple(values_tuples)}")
            self.cursor_db.execute(update_result, tuple(values_tuples))
            if self.cursor_db.rowcount == 1:
                self.logger.info(f"The job {results['job_id']} is updated")
            self.logger.info(f"Old results {old_results}, New Results {results}")
        self.close_connection()

    def append_error_string_in_running_job(self, job_id, error_message, **kwargs):
        self.init_connection(cursor_factory=None)
        running_jobs = f"{self.schema}.running_jobs"
        current_message = (f"SELECT cluster_id, error_history from {running_jobs} WHERE "
                           f"{running_jobs}.job_id = {job_id}")
        self.cursor_db.execute(current_message)
        cur_message = self.cursor_db.fetchone()
        error_message = "cluster{}".format(cur_message[0]) + error_message
        if cur_message[1] != "NA":
            error_message = error_message + ";\n" + cur_message[1]
        update_job = f"UPDATE {running_jobs} SET error_history = %s, interrupted = %s, finished=%s WHERE job_id = %s"
        self.cursor_db.execute(update_job, (error_message, True, False, job_id))
        if self.cursor_db.rowcount == 1:
            self.logger.info("The job {} is interrupted".format(job_id))
        self.close_connection()

    def get_lowest_job_id_with_hash(self, hash_value):
        self.init_connection(cursor_factory=None)
        avail_jobs = f"{self.schema}.avail_jobs"
        query = f"""SELECT job_id FROM {avail_jobs} WHERE hash_value = %s ORDER BY job_id ASC;"""

        # Execute the query
        self.cursor_db.execute(query, (hash_value,))

        # Fetch the result
        result = list(self.cursor_db.fetchall())
        lowest_job_id = None
        if result:
            lowest_job_id = result[0][0]
            self.logger.info(f"The lowest job_id with hash_value '{hash_value}' is {lowest_job_id}.")
            self.logger.info(f"The job {lowest_job_id} was not evaluated properly")
        else:
            self.logger.info(f"No job found with hash_value '{hash_value}'.")

        self.close_connection()
        return lowest_job_id

    def append_error_string_in_running_job2(self, job_id, error_message, **kwargs):
        self.init_connection(cursor_factory=None)
        running_jobs = f"{self.schema}.running_jobs"
        current_message = (f"SELECT cluster_id, error_history from {running_jobs} "
                           f"WHERE {running_jobs}.job_id = {job_id}")
        self.cursor_db.execute(current_message)
        cur_message = self.cursor_db.fetchone()
        error_message = f"cluster{cur_message[0]}" + error_message
        if cur_message[1] != "NA":
            error_message = error_message + ";\n" + cur_message[1]
        update_job = f"UPDATE {running_jobs} SET error_history = %s, interrupted = %s, finished=%s WHERE job_id = %s"
        self.cursor_db.execute(update_job, (error_message, False, False, job_id))
        if self.cursor_db.rowcount == 1:
            self.logger.info("The job {} is interrupted".format(job_id))
        self.close_connection()

    # def rename_all_jobs(self, DIR_PATH, LOGS_FOLDER, OPTIMIZER_FOLDER):
    #     self.init_connection()
    #     avail_jobs = "{}.avail_jobs".format(self.schema)
    #     select_job = "SELECT * FROM {0} WHERE {0}.dataset=\'synthetic_or\'".format(avail_jobs)
    #     self.cursor_db.execute(select_job)
    #     jobs_all = self.cursor_db.fetchall()
    #     for job in jobs_all:
    #         job_id = job['job_id']
    #         self.logger.info(job['hash_value'])
    #         self.job_description = job
    #         self.logger.info(print_dictionary(job))
    #         self.logger.info('old file name {}'.format(self.create_hash_value()))
    #         file_name_old = self.create_hash_value()
    #         old_log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "{}.log".format(file_name_old))
    #         old_opt_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, "{}".format(file_name_old))
    #
    #         # Change the current description
    #         self.job_description['dataset_params']['n_test_instances'] = self.job_description['dataset_params'][
    #                                                                          'n_train_instances'] * 10
    #         file_name_new = self.create_hash_value()
    #         new_log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "{}.log".format(file_name_new))
    #         new_opt_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, "{}".format(file_name_new))
    #         self.logger.info("log file exist {}".format(os.path.exists(old_log_path)))
    #         self.logger.info("opt file exist {}".format(os.path.exists(old_opt_path)))
    #
    #         # Rename the old optimizers and log files
    #         if os.path.exists(old_log_path):
    #             os.rename(old_log_path, new_log_path)
    #         if os.path.exists(old_opt_path):
    #             os.rename(old_opt_path, new_opt_path)
    #         self.logger.info("renaming {} to {}".format(old_opt_path, new_opt_path))
    #         self.logger.info('new file name {}'.format(self.create_hash_value()))
    #         update_job = "UPDATE {0} set hash_value = %s, dataset_params = %s where job_id =%s".format(avail_jobs)
    #         self.logger.info(update_job)
    #         d_param = json.dumps(self.job_description['dataset_params'])
    #         self.cursor_db.execute(update_job, (file_name_new, d_param, job_id))
    #     self.close_connection()

    def clone_job(self, cluster_id, fold_id):
        avail_jobs = "{}.avail_jobs".format(self.schema)
        running_jobs = "{}.running_jobs".format(self.schema)
        self.init_connection()
        job_desc = dict(self.job_description)
        job_desc["fold_id"] = fold_id
        query_job_id = job_desc["job_id"]
        learner, learner_params = job_desc["learner"], job_desc["learner_params"]
        if "dataset_type" in job_desc["dataset_params"].keys():
            dataset, value, value2 = (
                job_desc["dataset"],
                job_desc["dataset_params"]["dataset_type"],
                job_desc["dataset_params"]["n_objects"],
            )
            expression = "dataset_params->> '{}' = '{}'".format("dataset_type", value)
            expression = "{} AND dataset_params->> '{}' = '{}'".format(
                expression, "n_objects", value2
            )
        elif "year" in job_desc["dataset_params"].keys():
            dataset, value, value2 = (
                job_desc["dataset"],
                job_desc["dataset_params"]["year"],
                job_desc["dataset_params"]["n_objects"],
            )
            expression = "dataset_params->> '{}' = '{}'".format("year", value)
            expression = "{} AND dataset_params->> '{}' = '{}'".format(
                expression, "n_objects", value2
            )
        else:
            dataset = job_desc["dataset"]
            expression = True
        self.logger.info(
            "learner_params {} expression {}".format(learner_params, expression)
        )
        select_job = "SELECT * from {} where fold_id = {} AND learner = '{}' AND  dataset = '{}' AND {}".format(
            avail_jobs, fold_id, learner, dataset, expression
        )
        self.logger.info("Select job for duplication {}".format(select_job))
        self.cursor_db.execute(select_job)
        new_job_id = None
        if self.cursor_db.rowcount != 0:
            for query in self.cursor_db.fetchall():
                query = dict(query)
                self.logger.info("Duplicate job {}".format(query["job_id"]))
                if self.get_hash_value_for_job(job_desc) == self.get_hash_value_for_job(query):
                    new_job_id = query["job_id"]
                    self.logger.info(
                        "The job {} with fold {} already exist".format(
                            new_job_id, fold_id
                        )
                    )
                    break
        if new_job_id is None:
            del job_desc["job_id"]
            keys = list(job_desc.keys())
            columns = ", ".join(keys)
            index = keys.index("fold_id")
            keys[index] = str(fold_id)
            values_str = ", ".join(keys)
            insert_job = "INSERT INTO {0} ({1}) SELECT {2} FROM {0} where {0}.job_id = {3} RETURNING job_id".format(
                avail_jobs, columns, values_str, query_job_id
            )
            self.logger.info("Inserting job with new fold: {}".format(insert_job))
            self.cursor_db.execute(insert_job)
            new_job_id = self.cursor_db.fetchone()[0]

        self.logger.info(f"Job {new_job_id} with fold id {fold_id} updated/inserted")
        start = datetime.now()
        update_job = f"""UPDATE {avail_jobs} set job_allocated_time = %s, hash_value = %s WHERE job_id = %s"""
        self.cursor_db.execute(update_job, (start, job_desc["hash_value"], new_job_id))
        select_job = f"""SELECT * FROM {running_jobs} WHERE {running_jobs}.job_id = {new_job_id} FOR UPDATE"""
        self.cursor_db.execute(select_job)
        count_ = len(self.cursor_db.fetchall())
        if count_ == 0:
            insert_job = f"""INSERT INTO {running_jobs} (job_id, cluster_id ,finished, interrupted) 
                            VALUES ({new_job_id}, {cluster_id},FALSE, FALSE)"""
            self.cursor_db.execute(insert_job)
            if self.cursor_db.rowcount == 1:
                self.logger.info(f"The job {new_job_id} is inserted in running jobs")
        else:
            self.logger.info(f"Job with job_id {new_job_id} present in the updating and row locked")
            update_job = f"""UPDATE {running_jobs} set cluster_id = %s, interrupted = %s, 
                             finished = %s WHERE job_id = %s"""
            self.cursor_db.execute(update_job, (cluster_id, "FALSE", "FALSE", new_job_id))
            if self.cursor_db.rowcount == 1:
                self.logger.info(f"The job {new_job_id} is updated in running jobs")
        self.close_connection()

        return new_job_id

    def check_exists(self, job):
        if self.schema in [LEAKAGE_DETECTION, LEAKAGE_DETECTION_NEW, LEAKAGE_DETECTION_PADDING]:
            hash_value_new = self.get_hash_value_for_job_ild_check(job)
        else:
            hash_value_new = self.get_hash_value_for_job(job)
        self.logger.info(f"Hash Value New {hash_value_new}")
        for hash_value_existing in self.current_hash_values:
            if hash_value_existing == hash_value_new:
                return True
        return False

    def insert_new_jobs_openml(self, dataset=OPENML_PADDING_DATASET, max_job_id=7):
        self.init_connection()
        avail_jobs = f"{self.schema}.avail_jobs"
        # learners = [TABPNF, TABPNF]
        # select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' AND" \
        #             f" {avail_jobs}.job_id<={max_job_id} and {avail_jobs}.base_learner NOT IN {tuple(learners)} " \
        #             f"ORDER BY {avail_jobs}.job_id"
        select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' AND {avail_jobs}.job_id<={max_job_id} ORDER BY {avail_jobs}.job_id"
        self.cursor_db.execute(select_job)
        jobs_all = self.cursor_db.fetchall()
        self.logger.info(jobs_all)
        imbalances = [0.1, 0.3, 0.5]
        if dataset == OPENML_PADDING_DATASET:
            dataset_ids = list(get_openml_padding_datasets().keys())
        if dataset == OPENML_DATASET:
            dataset_ids = list(get_openml_datasets().keys())
        for job in jobs_all:
            job = dict(job)
            del job["job_id"]
            del job["job_allocated_time"]
            del job['job_end_time']
            job['evaluation_time'] = 0
            del job['hash_value']
            self.logger.info("###########################################################")
            self.logger.info(print_dictionary(job))
            detection_method = job['detection_method']
            base_learner = job["base_learner"]
            for dataset_id in dataset_ids:
                for imbalance in imbalances:
                    keys = list(job.keys())
                    values = list(job.values())
                    columns = ", ".join(list(job.keys()))
                    values_str = []
                    self.logger.info(f"Learner {base_learner} Detector Method {detection_method}")
                    for i, (key, val) in enumerate(zip(keys, values)):
                        if isinstance(val, dict):
                            if key == 'dataset_params':
                                val['dataset_id'] = dataset_id
                                val['imbalance'] = imbalance
                                self.logger.info(f"Dataset Params {val}")
                            val = json.dumps(val, cls=NpEncoder)
                        else:
                            val = str(val)
                        values_str.append(val)
                        if i == 0:
                            str_values = "%s"
                        else:
                            str_values = str_values + ", %s"
                    condition = self.check_exists(job)
                    if not condition:
                        insert_result = f"INSERT INTO {avail_jobs} ({columns}) VALUES ({str_values}) RETURNING job_id"
                        self.cursor_db.execute(insert_result, tuple(values_str))
                        id_of_new_row = self.cursor_db.fetchone()[0]
                        self.logger.info("Inserting results: {} {}".format(insert_result, values_str))
                        if self.cursor_db.rowcount == 1:
                            self.logger.info(f"Results inserted for the job {id_of_new_row}")
                        self.connection.commit()
                        hash_value_new = self.get_hash_value_for_job_ild_check(job)
                        self.current_hash_values.append(hash_value_new)
                    else:
                        self.logger.info(f"Job already exist")
        self.close_connection()
        if dataset == OPENML_PADDING_DATASET:
            self.insert_new_jobs_with_different_fold(dataset=OPENML_PADDING_DATASET, folds=4)

    def insert_detection_methods(self, dataset="openml_dataset"):
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format(self.schema)
        learners = [TABPFN_VAR, TABPFN_VAR]
        select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' AND {avail_jobs}.base_learner " \
                     f"NOT IN {tuple(learners)} ORDER  BY {avail_jobs}.job_id "
        select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' ORDER  BY {avail_jobs}.job_id "
        self.cursor_db.execute(select_job)
        jobs_all = self.cursor_db.fetchall()
        self.logger.info(jobs_all)

        cls_detection_methods = list(leakage_detection_methods.keys())
        mi_detection_method = re.sub(r'(?<!^)(?=[A-Z])', '_', ESTIMATED_MUTUAL_INFORMATION).lower()
        cls_detection_methods.remove(mi_detection_method)
        mi_detection_methods = [mi_detection_method]
        detection_methods = {MINE_MI_ESTIMATOR: mi_detection_methods, GMM_MI_ESTIMATOR: mi_detection_methods,
                             AUTO_GLUON: cls_detection_methods, AUTO_GLUON_STACK: cls_detection_methods,
                             TABPFN: cls_detection_methods, TABPFN_VAR: cls_detection_methods,
                             MULTI_LAYER_PERCEPTRON: cls_detection_methods, RANDOM_FOREST: cls_detection_methods}
        for job in jobs_all:
            job = dict(job)
            del job["job_id"]
            del job["job_allocated_time"]
            del job['job_end_time']
            job['evaluation_time'] = 0
            del job['hash_value']
            self.logger.info("###########################################################")
            self.logger.info(print_dictionary(job))
            base_learner = job["base_learner"]
            methods = detection_methods[base_learner]
            for detection_method in methods:
                job['detection_method'] = detection_method
                self.logger.info(f"Learner {base_learner} Detector Method {detection_method}")
                keys = list(job.keys())
                values = list(job.values())
                columns = ", ".join(list(job.keys()))
                values_str = []
                for i, (key, val) in enumerate(zip(keys, values)):
                    if isinstance(val, dict):
                        val = json.dumps(val, cls=NpEncoder)
                    else:
                        val = str(val)
                    values_str.append(val)
                    if i == 0:
                        str_values = "%s"
                    else:
                        str_values = str_values + ", %s"
                condition = self.check_exists(job)
                if not condition:
                    insert_result = f"INSERT INTO {avail_jobs} ({columns}) VALUES ({str_values}) RETURNING job_id"
                    self.cursor_db.execute(insert_result, tuple(values_str))
                    id_of_new_row = self.cursor_db.fetchone()[0]
                    self.logger.info("Inserting results: {} {}".format(insert_result, values_str))
                    if self.cursor_db.rowcount == 1:
                        self.logger.info(f"Results inserted for the job {id_of_new_row}")
                    self.connection.commit()
                    hash_value_new = self.get_hash_value_for_job_ild_check(job)
                    self.current_hash_values.append(hash_value_new)
                else:
                    self.logger.info(f"Job already exist")
        self.close_connection()

    def insert_new_jobs_with_different_fold(self, dataset="synthetic", folds=4):
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format(self.schema)
        select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' AND \
                        {avail_jobs}.fold_id =0 ORDER  BY {avail_jobs}.job_id"

        self.cursor_db.execute(select_job)
        jobs_all = self.cursor_db.fetchall()
        for job in jobs_all:
            job = dict(job)
            del job["job_id"]
            del job["job_allocated_time"]
            del job['job_end_time']
            job['evaluation_time'] = 0
            self.logger.info("###########################################################")
            self.logger.info(print_dictionary(job))
            for f_id in range(folds):
                job["fold_id"] = f_id + 1
                columns = ", ".join(list(job.keys()))
                values_str = []
                for i, val in enumerate(job.values()):
                    if isinstance(val, dict):
                        val = json.dumps(val)
                    # elif isinstance(val, str):
                    #   val = "\'{}\'".format(str(val))
                    else:
                        val = str(val)
                    values_str.append(val)
                    if i == 0:
                        values = "%s"
                    else:
                        values = values + ", %s"
                condition = self.check_exists(job)
                if not condition:
                    insert_result = f"INSERT INTO {avail_jobs} ({columns}) VALUES ({values}) RETURNING job_id"
                    self.logger.info("Inserting results: {} {}".format(insert_result, values_str))
                    self.cursor_db.execute(insert_result, tuple(values_str))
                    id_of_new_row = self.cursor_db.fetchone()[0]
                    if self.cursor_db.rowcount == 1:
                        self.logger.info("Results inserted for the job {}".format(id_of_new_row))
                    hash_value_new = self.get_hash_value_for_job(job)
                    self.current_hash_values.append(hash_value_new)
                    self.connection.commit()
        self.close_connection()

    def insert_new_jobs_different_configurations(self, dataset="synthetic", max_job_id=13):
        self.init_connection()
        avail_jobs = f"{self.schema}.avail_jobs"
        select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' AND " \
                     f"{avail_jobs}.fold_id =0 and {avail_jobs}.job_id<={max_job_id} ORDER  BY {avail_jobs}.job_id"

        self.cursor_db.execute(select_job)
        jobs_all = self.cursor_db.fetchall()
        features = np.arange(2, 21, step=2)
        classes = np.arange(2, 11, step=2)
        # flip_ys_fine = np.arange(0.0, 1.01, .01)
        flip_ys_broad = np.arange(0.0, 1.01, .1)
        for job in jobs_all:
            job = dict(job)
            del job["job_id"]
            del job["job_allocated_time"]
            del job['job_end_time']
            job['evaluation_time'] = 0
            self.logger.info("###########################################################")
            self.logger.info(print_dictionary(job))
            for n_classes in classes:
                for n_features in features:
                    for flip_y in flip_ys_broad:
                        keys = list(job.keys())
                        values = list(job.values())
                        columns = ", ".join(list(job.keys()))
                        values_str = []
                        for i, (key, val) in enumerate(zip(keys, values)):
                            if isinstance(val, dict):
                                if key == 'dataset_params':
                                    val['n_classes'] = n_classes
                                    val['n_features'] = n_features
                                    if dataset == SYNTHETIC_DATASET:
                                        val['flip_y'] = flip_y.round(2)
                                    if dataset == SYNTHETIC_DISTANCE_DATASET:
                                        val['noise'] = flip_y.round(2)

                                val = json.dumps(val, cls=NpEncoder)
                            else:
                                val = str(val)
                            values_str.append(val)
                            if i == 0:
                                str_values = "%s"
                            else:
                                str_values = str_values + ", %s"
                        condition = self.check_exists(job)
                        if not condition:
                            insert_result = f"INSERT INTO {avail_jobs} ({columns}) VALUES ({str_values}) RETURNING job_id"
                            self.cursor_db.execute(insert_result, tuple(values_str))
                            id_of_new_row = self.cursor_db.fetchone()[0]
                            self.logger.info("Inserting results: {} {}".format(insert_result, values_str))
                            if self.cursor_db.rowcount == 1:
                                self.logger.info(f"Results inserted for the job {id_of_new_row}")
                            hash_value_new = self.get_hash_value_for_job(job)
                            self.current_hash_values.append(hash_value_new)
                            self.connection.commit()
                        else:
                            self.logger.info(f"Job already exist")
        self.close_connection()

    def insert_new_jobs_imbalanced(self, dataset="synthetic_imbalanced", max_job_id=13):
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format(self.schema)
        select_job = f"SELECT * FROM {avail_jobs} WHERE {avail_jobs}.dataset='{dataset}' AND " \
                     f"{avail_jobs}.fold_id =0 and {avail_jobs}.job_id<={max_job_id} ORDER  BY {avail_jobs}.job_id"

        self.cursor_db.execute(select_job)
        jobs_all = self.cursor_db.fetchall()
        print(jobs_all)
        n_features = 5
        classes = [2, 5]
        flip_ys_broad = np.arange(0.0, 1.01, .1)
        imbalance_dictionary = {2: np.arange(0.05, 0.51, .05), 5: np.arange(0.02, 0.21, .02)}
        gen_types = {2: ['single'], 5: GEN_TYPES}
        for job in jobs_all:
            job = dict(job)
            del job["job_id"]
            del job["job_allocated_time"]
            del job['job_end_time']
            job['evaluation_time'] = 0
            self.logger.info("###########################################################")
            self.logger.info(print_dictionary(job))
            for n_classes in classes:
                self.logger.info("###########################################################")
                for flip_y in flip_ys_broad:
                    imbalances = imbalance_dictionary[n_classes]
                    for gen_type in gen_types[n_classes]:
                        if gen_type == 'multiple' and n_classes == 2:
                            self.logger.info(f"Skipping configuration for n_classes {n_classes} gen_type {gen_type}")
                            continue
                        for imbalance in imbalances[::-1]:
                            self.logger.info(f"Inserting job with gen_type {gen_type}, imbalance {imbalance}, "
                                             f"flip_y {flip_y} and n_classes {n_classes}")

                            keys = list(job.keys())
                            values = list(job.values())
                            columns = ", ".join(list(job.keys()))
                            values_str = []
                            samples_per_class = generate_samples_per_class(n_classes, samples=1000, imbalance=imbalance,
                                                                           gen_type=gen_type, logger=self.logger)
                            for i, (key, val) in enumerate(zip(keys, values)):
                                if isinstance(val, dict):
                                    if key == 'dataset_params':
                                        val['n_classes'] = n_classes
                                        val['n_features'] = n_features
                                        val['samples_per_class'] = samples_per_class
                                        if dataset in [SYNTHETIC_DATASET, SYNTHETIC_IMBALANCED_DATASET]:
                                            val['flip_y'] = flip_y.round(2)
                                        if dataset in [SYNTHETIC_DISTANCE_DATASET,
                                                       SYNTHETIC_DISTANCE_IMBALANCED_DATASET]:
                                            val['noise'] = flip_y.round(2)
                                        val['imbalance'] = imbalance.round(2)
                                        val['gen_type'] = gen_type
                                        self.logger.info(f"Dataset Params {val}")
                                    val = json.dumps(val, cls=NpEncoder)
                                else:
                                    val = str(val)
                                values_str.append(val)
                                if i == 0:
                                    str_values = "%s"
                                else:
                                    str_values = str_values + ", %s"
                            condition = self.check_exists(job)
                            if not condition:
                                insert_result = f"INSERT INTO {avail_jobs} ({columns}) VALUES ({str_values}) RETURNING job_id"
                                self.cursor_db.execute(insert_result, tuple(values_str))
                                id_of_new_row = self.cursor_db.fetchone()[0]
                                self.logger.info("Inserting results: {} {}".format(insert_result, values_str))
                                if self.cursor_db.rowcount == 1:
                                    self.logger.info(f"Results inserted for the job {id_of_new_row}")
                                hash_value_new = self.get_hash_value_for_job(job)
                                self.current_hash_values.append(hash_value_new)
                                self.connection.commit()
                            else:
                                self.logger.info(f"Job already exist")
        self.close_connection()
