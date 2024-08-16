import inspect
import logging
import multiprocessing
import os
import random

import numpy as np
import sklearn
import tensorflow as tf
import torch
from packaging import version
from sklearn.utils import check_random_state
from skopt.space import Real, Categorical, Integer


def create_search_space(hp_ranges, logger):
    def isint(v):
        return type(v) is int

    def isfloat(v):
        return type(v) is float

    def isbool(v):
        return type(v) is bool

    def isstr(v):
        return type(v) is str

    search_space = {}
    for key, value in hp_ranges.items():
        logger.info(f"Before key {key} value {value}")
        if version.parse(sklearn.__version__) < version.parse("0.25.0"):
            if key == "criterion" and "squared_error" in value:
                value = ["friedman_mse", "mse"]
        if isint(value[0]) and isint(value[1]):
            search_space[key] = Integer(value[0], value[1])
        if isfloat(value[0]) and isfloat(value[1]):
            if len(value) == 3:
                search_space[key] = Real(value[0], value[1], prior=value[2])
        if (isbool(value[0]) and isbool(value[1])) or (isstr(value[0]) and isstr(value[1])):
            search_space[key] = Categorical(value)
        logger.info(f"key {key} value {value}")
    return search_space


def setup_logging(log_path=None, level=logging.INFO):
    """Function setup as many logging for the experiments."""
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "logs", "logs.log")

    # log = logging.getLogger()  # root logger
    # for hdlr in log.handlers[:]:  # remove all old handlers
    #    log.removeHandler(hdlr)
    #
    # fileh = logging.FileHandler(log_path, 'a')
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
    # fileh.setFormatter(formatter)
    # fileh.setLevel(level)
    # log.addHandler(fileh)
    # log.setLevel(level)
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("SetupLogging")  # root logger
    logger.info("log file path: {}".format(log_path))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses INFO, WARNING, and ERROR logs
    # Additional TensorFlow setting to disable GPU usage explicitly
    tf.config.set_visible_devices([], "GPU")
    logging.captureWarnings(False)
    import warnings

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("pytorch").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def setup_random_seed(random_state=1234):
    # logger.info('Seed value: {}'.format(seed))
    logger = logging.getLogger("Setup Logging")
    random_state = check_random_state(random_state)

    seed = random_state.randint(2**31, dtype="uint32")
    torch.manual_seed(seed)
    logger.info(f"Total number of torch threads {torch.get_num_threads()}")
    if torch.get_num_threads() <= 2:
        n_cpus = 1
    else:
        n_cpus = torch.get_num_threads() - 2
        if "pc2" in os.environ["HOME"]:
            n_cpus = 4
    logger.info(f"Torch threads set {n_cpus}")

    torch.set_num_threads(n_cpus)
    tf.random.set_seed(seed)

    seed = random_state.randint(2**31, dtype="uint32")
    np.random.seed(seed)
    random.seed(seed)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    devices = tf.config.list_physical_devices("GPU")
    logger.info("Keras Devices {}".format(devices))
    n_gpus = len(devices)
    logger.info("Keras GPU {}".format(n_gpus))
    if n_gpus == 0:
        # Limiting CPU usage
        cpu_count = multiprocessing.cpu_count()
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        if cpu_count > 2:
            # TensorFlow doesn't directly allow setting CPU count, but you can control parallelism
            # For limiting CPU, consider setting parallelism threads as shown above
            pass
    else:
        # Configuring GPU options
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    torch_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Torch GPU device {}".format(torch_gpu))
