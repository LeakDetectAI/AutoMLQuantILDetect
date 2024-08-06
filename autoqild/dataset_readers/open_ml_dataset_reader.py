import logging
from abc import ABCMeta

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from .utils import *

__all__ = ['OpenMLTimingDatasetReader', 'OpenMLPaddingDatasetReader']



class OpenMLTimingDatasetReader(metaclass=ABCMeta):
    def __init__(self, dataset_id: int, imbalance: float, create_datasets=True, random_state=None, **kwargs):
        self.logger = logging.getLogger(OpenMLTimingDatasetReader.__name__)
        self.dataset_id = dataset_id
        self.imbalance = imbalance
        self.random_state = check_random_state(random_state)
        self.correct_class = 'Correctly_formatted_PKCS#1_PMS_message'
        self.vulnerable_classes = []
        self.__read_dataset__()
        self.__clean_up_dataset__()
        if create_datasets:
            self.__create_leakage_datasets__()

    def __read_dataset__(self):
        self.dataset = openml.datasets.get_dataset(self.dataset_id, download_data=True)
        # Access the dataset information
        self.data_frame_raw, _, _, self.attribute_names = self.dataset.get_data(dataset_format='dataframe')
        self.attribute_names.remove(LABEL_COL)
        self.dataset_dictionary = {}
        if self.correct_class not in self.data_frame_raw[LABEL_COL].unique():
            raise ValueError(f'Dataframe is does not contain correct class {self.correct_class}')
        self.logger.info(f"Class Labels unformulated {list(self.data_frame_raw[LABEL_COL].unique())}")
        description = self.dataset.description
        vulnerable_classes_str = description.split('\n')[-1].split("vulnerable_classes ")[-1]
        vulnerable_classes_str = vulnerable_classes_str.strip('[]')
        self.vulnerable_classes = [s.strip() for s in vulnerable_classes_str.split(',')]
        self.n_features = len(self.dataset.features) - 1
        self.fold_id = int(description.split('\n')[-2].split("fold_id ")[-1])
        self.delay = int(description.split('Bleichenbacher Timing Attack: ')[-1].split(" micro seconds")[0])

    def __clean_up_dataset__(self):
        categorical_columns = self.data_frame_raw.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            if column != LABEL_COL:
                self.data_frame_raw[column] = label_encoder.fit_transform(self.data_frame_raw[column])

        self.data_frame_raw[LABEL_COL] = self.data_frame_raw[LABEL_COL].apply(lambda x: clean_class_label(x))
        self.correct_class = clean_class_label(self.correct_class)
        self.vulnerable_classes = [clean_class_label(s) for s in self.vulnerable_classes]
        labels = list(self.data_frame_raw[LABEL_COL].unique())
        labels.sort()
        self.logger.info(f"Class Labels formatted {labels}")
        self.logger.info(f"Correct Padding {self.correct_class}")
        self.logger.info(f"Vulnerable Padding {self.vulnerable_classes}")

        labels.remove(self.correct_class)
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        self.label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_) + 1))
        self.label_mapping = {**{self.correct_class: 0}, **self.label_mapping}
        self.inverse_label_mapping = dict((v, k) for k, v in self.label_mapping.items())
        self.n_labels = len(self.label_mapping)
        self.data_frame = pd.DataFrame.copy(self.data_frame_raw)
        self.data_frame[LABEL_COL].replace(self.label_mapping, inplace=True)
        self.data_frame = self.data_frame.fillna(value=-1)

    def __create_leakage_datasets__(self):
        self.dataset_dictionary = {}
        for j, label in self.inverse_label_mapping.items():
            if label == self.correct_class:
                continue
            else:
                self.dataset_dictionary[label] = self.get_data(class_label=j)

    def get_data(self, class_label=1):
        df = pd.DataFrame.copy(self.data_frame)
        p = [0, class_label]
        df = df[df.label.isin(p)]
        df[LABEL_COL].replace([class_label], 1, inplace=True)
        X, y = df[self.attribute_names].values, df[LABEL_COL].values.flatten()
        X, y = self.get_sampled_imbalanced_data(X, y)
        return X, y

    def get_sampled_imbalanced_data(self, X, y):
        if self.imbalance < 0.5:
            # total_instances = X.shape[0]
            n_0 = len(np.where(y == 0)[0])
            n_1 = int(n_0 * (self.imbalance / (1 - self.imbalance)))
            self.logger.info(f"Before processing----ratio {n_1 / n_0} p {self.imbalance}, n_0 {n_0}, n_1 {n_1}----")
            ind0 = np.where(y == 0)[0]
            ind1 = self.random_state.choice(np.where(y == 1)[0], n_1)
            if n_1 < 200:
                ind0 = np.concatenate((ind0, ind0))
                ind1 = self.random_state.choice(np.where(y == 1)[0], 2 * n_1)
            indx = np.concatenate((ind0, ind1))
            self.random_state.shuffle(indx)
            X, y = X[indx], y[indx]
            n_0 = len(np.where(y == 0)[0])
            n_1 = len(np.where(y == 1)[0])
            self.logger.info(f"After processing----ratio {n_1 / n_0} p {self.imbalance}, n_0 {n_0}, n_1 {n_1}----")
        return X, y


class OpenMLPaddingDatasetReader(OpenMLTimingDatasetReader):
    def __init__(self, dataset_id: int, imbalance: float, create_datasets=True, random_state=None, **kwargs):
        super().__init__(dataset_id=dataset_id, imbalance=imbalance, create_datasets=create_datasets,
                         random_state=random_state, **kwargs)
        self.logger = logging.getLogger(OpenMLPaddingDatasetReader.__name__)

        if create_datasets:
            self.__create_leakage_datasets__()

    def __read_dataset__(self):
        self.dataset = openml.datasets.get_dataset(self.dataset_id, download_data=True)
        # Access the dataset information
        self.data_frame_raw, _, _, self.attribute_names = self.dataset.get_data(dataset_format='dataframe')
        self.attribute_names.remove(LABEL_COL)
        self.dataset_dictionary = {}
        if self.correct_class not in self.data_frame_raw[LABEL_COL].unique():
            raise ValueError(f'Dataframe is does not contain correct class {self.correct_class}')
        self.logger.info(f"Class Labels unformulated {list(self.data_frame_raw[LABEL_COL].unique())}")
        description = self.dataset.description
        vulnerable_classes_str = description.split('\n')[-1].split("vulnerable_classes ")[-1]
        vulnerable_classes_str = vulnerable_classes_str.strip('[]')
        self.vulnerable_classes = [s.strip() for s in vulnerable_classes_str.split(',')]
        self.n_features = len(self.dataset.features) - 1
        self.server = self.dataset.name.split('padding-attack-dataset-')[-1]
