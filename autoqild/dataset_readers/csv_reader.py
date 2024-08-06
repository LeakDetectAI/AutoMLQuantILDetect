import logging
import os
from abc import ABCMeta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from .utils import clean_class_label
from ..utilities import print_dictionary

sns.set(color_codes=True)
plt.style.use('default')
LABEL_COL = 'label'
MISSING_CCS_FIN = 'missing_ccs_fin'


def str2bool(v):
    if int(v) > 0:
        v = 'true'
    return str(v).lower() in ("yes", "true", "t", "1")


class CSVReader(metaclass=ABCMeta):
    def __init__(self, folder: str, preprocessing='replace', **kwargs):
        self.logger = logging.getLogger(CSVReader.__name__)
        self.dataset_folder = folder
        self.f_file = os.path.join(self.dataset_folder, "Feature Names.csv")
        self.df_file = os.path.join(self.dataset_folder, "Features.csv")
        self.v_classes_file = os.path.join(self.dataset_folder, "Vulnerable Classes.pickle")
        self.preprocessing = preprocessing
        self.ccs_fin_array = [False]
        self.correct_class = "Correctly Formatted Pkcs#1 Pms Message"
        self.data_frame = None
        self.__load_dataset__()

    def __load_dataset__(self):
        if not os.path.exists(self.df_file):
            raise ValueError(f"No such file or directory: {self.df_file}")
        if not os.path.exists(self.f_file):
            raise ValueError(f"No such file or directory: {self.f_file}")
        self.data_frame = pd.read_csv(self.df_file, index_col=0)

        if LABEL_COL not in self.data_frame.columns:
            error_string = 'Dataframe does not contain label columns'
            if self.data_frame.shape[0] == 0:
                raise ValueError(f'Dataframe is empty and {error_string}')
        else:
            df = pd.DataFrame.copy(self.data_frame)
            df[LABEL_COL] = df[LABEL_COL].apply(lambda x: ' '.join(x.split('_')).title())
            if self.correct_class not in df[LABEL_COL].unique():
                raise ValueError(f'Dataframe is does not contain correct class {self.correct_class}')
        self.data_frame[LABEL_COL] = self.data_frame[LABEL_COL].apply(lambda x: clean_class_label(x))
        labels = list(self.data_frame[LABEL_COL].unique())
        labels.sort()
        labels.remove(self.correct_class)

        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        self.label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_) + 1))
        self.label_mapping = {**{self.correct_class: 0}, **self.label_mapping}
        self.inverse_label_mapping = dict((v, k) for k, v in self.label_mapping.items())
        self.n_labels = len(self.label_mapping)
        if len(self.data_frame[MISSING_CCS_FIN].unique()) == 1:
            del self.data_frame[MISSING_CCS_FIN]
        self.data_raw = pd.DataFrame.copy(self.data_frame)
        self.data_frame[LABEL_COL].replace(self.label_mapping, inplace=True)
        self.logger.info(f"Label Mapping {print_dictionary(self.label_mapping)}")
        self.logger.info(f"Inverse Label Mapping {print_dictionary(self.inverse_label_mapping)}")

        if self.preprocessing == 'replace':
            self.data_frame = self.data_frame.fillna(value=-1)
        elif self.preprocessing == 'remove':
            cols = [c for c in self.data_frame.columns if 'msg1' not in c or 'msg5' not in c]
            self.data_frame = self.data_frame[cols]
            self.data_frame = self.data_frame.fillna(value=-1)
        self.features = pd.read_csv(self.f_file, index_col=0)
        self.feature_names = self.features['machine'].values.flatten()
        if MISSING_CCS_FIN in self.data_frame.columns:
            self.data_frame[MISSING_CCS_FIN] = self.data_frame[MISSING_CCS_FIN].apply(str2bool)
            self.ccs_fin_array = list(self.data_frame[MISSING_CCS_FIN].unique())
        df = pd.DataFrame.copy(self.data_frame)
        df[LABEL_COL].replace(self.inverse_label_mapping, inplace=True)
        if MISSING_CCS_FIN in self.data_frame.columns:
            df = pd.DataFrame(df[[LABEL_COL, MISSING_CCS_FIN]].value_counts().sort_index())
            df.reset_index(inplace=True)
            df.rename({0: 'Frequency'}, inplace=True, axis='columns')
            df.sort_values(by=[MISSING_CCS_FIN, LABEL_COL], inplace=True)
            f_vals = df.loc[df[LABEL_COL] == self.correct_class][[MISSING_CCS_FIN, 'Frequency']].values
            vals = dict(zip(f_vals[:, 0], f_vals[:, 1]))

            def div(row, val):
                return row['Frequency'] / val

            df['ratio_1_0'] = df.apply(
                lambda row: div(row, vals[True]) if str2bool(row.missing_ccs_fin) else div(row, vals[False]), axis=1)
            fname = os.path.join(self.dataset_folder, "label_frequency.csv")
            df.to_csv(fname)
        else:
            df = pd.DataFrame(df[LABEL_COL].value_counts().sort_index())
            df.reset_index(inplace=True)
            df.rename({0: 'Frequency'}, inplace=True, axis='columns')
            df.sort_values(by=[LABEL_COL], inplace=True)
            df['ratio_1_0'] = df[LABEL_COL].value_counts() / len(df.loc[df[LABEL_COL] == self.correct_class])
            fname = os.path.join(self.dataset_folder, "label_frequency.csv")
            df.to_csv(fname)
        self.label_frequency = df

    def get_data_class_label(self, class_label=1, missing_ccs_fin=False, sample=True):
        if sample:
            ratios = self.data_frame[LABEL_COL].value_counts(normalize=True)
            self.data_frame = self.data_frame.groupby(LABEL_COL).apply(
                lambda x: x.sample(int(ratios[x.name] * 20000), replace=True)).reset_index(drop=True)

        if MISSING_CCS_FIN in self.data_frame.columns:
            df = self.data_frame[self.data_frame[MISSING_CCS_FIN] == missing_ccs_fin]
        else:
            df = self.data_frame
        if class_label == 0:
            x, y = self.get_data(df)
        else:
            p = [0, class_label]
            df = df[df.label.isin(p)]
            df[LABEL_COL].replace([class_label], 1, inplace=True)
            x, y = df[self.feature_names].values, df[LABEL_COL].values.flatten()
        return x, y

    def get_data(self, df):
        x, y = df[self.feature_names].values, df[LABEL_COL].values.flatten()
        return x, y
