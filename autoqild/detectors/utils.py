import hashlib
import re

from netcal.binning import IsotonicRegression, HistogramBinning
from netcal.scaling import LogisticCalibration, BetaCalibration, TemperatureScaling
from sklearn.metrics import confusion_matrix, accuracy_score

from ..utilities import *

__all__ = ['mi_estimation_metrics', 'classification_leakage_detection_methods', 'mi_leakage_detection_methods',
           'leakage_detection_methods', 'calibrators', 'calibrator_params']
mi_estimation_metrics = {
    ACCURACY: accuracy_score,
    CONFUSION_MATRIX: confusion_matrix,
    MID_POINT_MI_ESTIMATION: mid_point_mi,
    LOG_LOSS_MI_ESTIMATION: log_loss_estimation,
    LOG_LOSS_MI_ESTIMATION_ISOTONIC_REGRESSION: log_loss_estimation,
    LOG_LOSS_MI_ESTIMATION_PLATT_SCALING: log_loss_estimation,
    LOG_LOSS_MI_ESTIMATION_BETA_CALIBRATION: log_loss_estimation,
    LOG_LOSS_MI_ESTIMATION_TEMPERATURE_SCALING: log_loss_estimation,
    LOG_LOSS_MI_ESTIMATION_HISTOGRAM_BINNING: log_loss_estimation,
    PC_SOFTMAX_MI_ESTIMATION: pc_softmax_estimation
}

classification_leakage_detection_methods = {
    PAIRED_TTEST_RANDOM: ACCURACY,
    PAIRED_TTEST: ACCURACY,
    FISHER_EXACT_TEST_MEAN: CONFUSION_MATRIX,
    FISHER_EXACT_TEST_MEDIAN: CONFUSION_MATRIX,
}
mi_leakage_detection_methods = {}
for value in [ESTIMATED_MUTUAL_INFORMATION, MID_POINT_MI_ESTIMATION, LOG_LOSS_MI_ESTIMATION,
              LOG_LOSS_MI_ESTIMATION_ISOTONIC_REGRESSION, LOG_LOSS_MI_ESTIMATION_PLATT_SCALING,
              LOG_LOSS_MI_ESTIMATION_BETA_CALIBRATION, LOG_LOSS_MI_ESTIMATION_TEMPERATURE_SCALING,
              LOG_LOSS_MI_ESTIMATION_HISTOGRAM_BINNING, PC_SOFTMAX_MI_ESTIMATION]:
    key = re.sub(r'(?<!^)(?=[A-Z])', '_', value).lower()
    key = key.replace('m_i', "mi")
    mi_leakage_detection_methods[key] = value
leakage_detection_methods = {**classification_leakage_detection_methods, **mi_leakage_detection_methods}
leakage_detection_names = {}
for key in leakage_detection_methods.keys():
    hash_object = hashlib.sha1()
    hash_object.update(key.encode())
    leakage_detection_names[key] = str(hash_object.hexdigest())[:8]
calibrators = {ISOTONIC_REGRESSION: IsotonicRegression,
               PLATT_SCALING: LogisticCalibration,
               HISTOGRAM_BINNING: HistogramBinning,
               BETA_CALIBRATION: BetaCalibration,
               TEMPERATURE_SCALING: TemperatureScaling}
calibrator_params = {ISOTONIC_REGRESSION: {'detection': False, 'independent_probabilities': False},
                     PLATT_SCALING: {'temperature_only': False, 'method': 'mle'},
                     HISTOGRAM_BINNING: {'detection': False, 'independent_probabilities': False},
                     BETA_CALIBRATION: {'detection': False, 'independent_probabilities': False},
                     TEMPERATURE_SCALING: {'detection': False, 'independent_probabilities': False}}
