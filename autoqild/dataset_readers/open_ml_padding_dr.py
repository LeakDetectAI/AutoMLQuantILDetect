import logging

import openml

from .open_ml_timming_dr import OpenMLTimingDatasetReader, LABEL_COL


class OpenMLPaddingDatasetReader(OpenMLTimingDatasetReader):
    """
    Reader for OpenML datasets related to leakages with respect to the error codes for each padding manipulation.

    This class extends `OpenMLTimingDatasetReader` and is tailored for datasets extracted from network traces
    exploiting error codes in the network traces to perform side-channel attacks, such as the Bleichenbacher
    timing attack. It reads, cleans, and processes the dataset, and provides methods to create datasets with
    class imbalance to simulate attack scenarios.

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

    dataset : openml.datasets.OpenMLDataset
        The OpenML dataset object.

    data_frame_raw : pandas.DataFrame
        The raw DataFrame containing the dataset.

    attribute_names : list of str
        List of attribute names (features) in the dataset.

    dataset_dictionary : dict
        A dictionary where keys are vulnerable class labels and values are tuples of (X, y) for the respective classes.

    n_features : int
        Number of features in the dataset.

    server : str
        The server associated with the padding attack dataset.

    vulnerable_classes : list of str
        List of class labels representing vulnerable (incorrectly formatted) messages.

    correct_class : str
        The correct class label, representing correctly formatted messages.

    Private Methods
    ---------------
    __read_dataset__()
        Reads the dataset from OpenML and extracts relevant information. This method fetches the dataset using the
        OpenML API, extracts the raw data, and processes the dataset description to retrieve vulnerable class labels,
        number of features, and server information.

    __create_leakage_datasets__()
        Creates separate datasets for each class by selecting only the samples that belong to the correct class and one
        vulnerable class at a time.

    __clean_up_dataset__()
        Cleans and preprocesses the dataset. This method encodes categorical columns, formats class labels, fills
        missing values, and convert class label strings to integer values.
    """
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

    def __create_leakage_datasets__(self):
        super().__create_leakage_datasets__()

    def __clean_up_dataset__(self):
        super().__clean_up_dataset__()

    def get_data(self, class_label=1):
        """
        Retrieves data for a specific class label.

        Parameters
        ----------
        class_label : int, default=1
            The class label for which to retrieve the data.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.
        """
        super().get_data(class_label=class_label)

    def get_sampled_imbalanced_data(self, X, y):
        """
        Creates an imbalanced dataset by sampling from the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Feature matrix after applying sampling to create imbalance.

        y : array-like of shape (n_samples,)
            Target vector after applying sampling to create imbalance.
        """
        super().get_sampled_imbalanced_data(X=X, y=y)
