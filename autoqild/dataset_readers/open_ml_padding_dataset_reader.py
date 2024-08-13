import logging

import openml

from autoqild import OpenMLTimingDatasetReader, LABEL_COL


class OpenMLPaddingDatasetReader(OpenMLTimingDatasetReader):
    def __init__(self, dataset_id: int, imbalance: float, create_datasets=True, random_state=None, **kwargs):
        super().__init__(dataset_id=dataset_id, imbalance=imbalance, create_datasets=create_datasets,
                         random_state=random_state, **kwargs)
        """
            Reader for OpenML datasets related to leakages with respect to the error codes for each padding 
            manipulation.

            This class extends OpenMLTimingDatasetReader and is tailored for datasets extracted from network traces exploiting
            error codes in the network traces to perform the side channel attacks, such as the Bleichenbacher timing attack. 
            It reads, cleans, and processes the dataset, and provides methods to create datasets with class imbalance to simulate attack scenarios.

            Parameters
            ----------
            dataset_id : int
                The ID of the OpenML dataset.

            imbalance : float
                The ratio of the number of minority class samples to the number of majority class samples. 
                Must be between 0 and 1.

            create_datasets : bool, default=True
                If True, creates leakage datasets during initialization.

            random_state : int or RandomState instance, optional
                Random state for reproducibility.

            **kwargs : dict
                Additional keyword arguments.

            Attributes
            ----------
            logger : logging.Logger
                Logger instance for logging information.

            server : str
                The server associated with the padding attack dataset.
        """
        self.logger = logging.getLogger(OpenMLPaddingDatasetReader.__name__)

        if create_datasets:
            self.__create_leakage_datasets__()

    def __read_dataset__(self):
        """
            Reads the dataset from OpenML and extracts relevant information.
            This method fetches the dataset using the OpenML API, extracts the raw data, and processes the dataset
            description to retrieve vulnerable class labels, number of features, and server information.
        """
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
