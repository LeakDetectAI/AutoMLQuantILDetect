"""Generates synthetic datasets by instroducing noise with reducing the distance between gaussians of each class,
simulating different distributions."""
import logging
from abc import ABCMeta

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from sklearn.utils import check_random_state

from .utils import FACTOR, pdf
from ..utilities import *

__all__ = [`SyntheticDatasetGeneratorDistance`]


class SyntheticDatasetGeneratorDistance(metaclass=ABCMeta):
    """
    Generator for synthetic datasets with a focus on generating data with varying class distances.

    This class generates synthetic datasets by adjusting the distance between class distributions, allowing
    for the simulation of scenarios with varying levels of overlap between classes. It is designed to help
    in scenarios like testing classifiers on datasets with controlled class separability, with a focus on
    distance-based variations.

    Parameters
    ----------
    n_classes : int, default=2
        Number of classes in the generated dataset.

    n_features : int, default=2
        Number of features in the generated dataset.

    samples_per_class : int or dict, default=500
        Number of samples per class. If an integer is provided, it is assumed that all classes have the same
        number of samples. If a dictionary is provided, the keys should be class labels and values should be
        the number of samples for each class.

    noise : float, default=0.1
        The level of noise to apply when generating class distributions, affecting the overlap between classes.

    random_state : int or RandomState instance, default=42
        Random state for reproducibility.

    fold_id : int, default=0
        Fold ID used for random seed generation.

    imbalance : float, default=0.0
        Proportion of the minority class in the dataset. Must be between 0 and 1.

    gen_type : str, default=`single`
        Type of generation process. It can be used to modify the dataset generation method.

    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    n_classes : int
        Number of classes in the generated dataset.

    n_features : int
        Number of features in the generated dataset.

    random_state : RandomState instance
        Random state instance for reproducibility.

    fold_id : int
        Fold ID used for random seed generation.

    means : dict
        Dictionary storing the mean vectors for each class.

    covariances : dict
        Dictionary storing the covariance matrices for each class.

    seeds : dict
        Dictionary storing the random seeds used for generating each class.

    samples_per_class : dict
        Dictionary storing the number of samples for each class.

    imbalance : float
        Proportion of the minority class in the dataset.

    gen_type : str
        Type of generation process.

    n_instances : int
        Total number of instances in the generated dataset.

    class_labels : numpy.ndarray
        Array of class labels.

    y_prob : dict
        Dictionary storing the probability of each class.

    noise : float
        The level of noise applied when generating class distributions.

    logger : logging.Logger
        Logger instance for logging information.

    Private Methods
    ---------------
    __generate_cov_means__():
        Generate the mean vectors and covariance matrices for each class.
        This method creates a random orthogonal matrix and generates a positive semi-definite covariance matrix.
        It then calculates the mean vector for each class.
    """
    def __init__(self, n_classes=2, n_features=2, samples_per_class=500, noise=0.1, random_state=42, fold_id=0,
                 imbalance=0.0, gen_type=`single`, **kwargs):
        self.n_classes = n_classes
        self.n_features = n_features
        self.random_state = check_random_state(random_state)
        self.fold_id = fold_id
        self.means = {}
        self.covariances = {}
        self.seeds = {}
        if isinstance(samples_per_class, int):
            self.samples_per_class = dict.fromkeys(np.arange(n_classes), samples_per_class)
        elif isinstance(samples_per_class, dict):
            self.samples_per_class = {}
            for key in samples_per_class.keys():
                self.samples_per_class[int(key)] = samples_per_class.get(key)
        else:
            raise ValueError("Samples per class is not defined properly")
        self.imbalance = imbalance
        self.gen_type = gen_type
        self.n_instances = sum(self.samples_per_class.values())
        self.class_labels = np.arange(self.n_classes)
        self.y_prob = {}
        self.ent_y = None
        self.noise = noise
        self.__generate_cov_means__()
        self.logger = logging.getLogger(SyntheticDatasetGeneratorDistance.__name__)

    def __generate_cov_means__(self):
        seed = self.random_state.randint(2 ** 31, dtype="uint32") + self.fold_id
        rs = np.random.RandomState(seed=seed)
        Q = ortho_group.rvs(dim=self.n_features)
        S = np.diag(np.diag(rs.rand(self.n_features, self.n_features)))
        cov = np.dot(np.dot(Q, S), np.transpose(Q))
        for k_class in self.class_labels:
            # A = rs.rand(n_features, n_features)
            # matrix1 = np.matmul(A, A.transpose())
            # positive semi-definite matrix
            seed = self.random_state.randint(2 ** 31, dtype="uint32") + self.fold_id
            mean = np.ones(self.n_features) + (k_class * FACTOR) * (1 - self.noise)
            self.means[k_class] = mean
            self.covariances[k_class] = cov
            self.seeds[k_class] = seed
            self.y_prob[k_class] = self.samples_per_class[k_class] / self.n_instances
        # print(self.y_prob)
        # print(self.flip_y_prob)

    def get_prob_dist_x_given_y(self, k_class):
        """
        Get the multivariate normal distribution for a given class.

        Parameters
        ----------
        k_class : int
            The class label for which to get the distribution.

        Returns
        -------
        scipy.stats._multivariate.multivariate_normal_frozen
            The multivariate normal distribution for the given class.
        """
        return multivariate_normal(mean=self.means[k_class], cov=self.covariances[k_class],
                                   seed=self.seeds[k_class])

    def get_prob_fn_margx(self):
        """
        Get the marginal probability distribution function for the input data.

        Returns
        -------
        marg_x: lambda function
            A function that computes the marginal probability for the input data.
        """
        marg_x = lambda x: np.array([self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x)
                                     for k_class in self.class_labels])
        return marg_x

    def get_prob_x_given_y(self, X, class_label):
        """
       Get the probability of X given a specific class label.

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)
           Input data.

       class_label : int
           The class label for which to compute the probability.

       Returns
       -------
       prob_x_given_y: array-like
           The probability of X given the class label.
        """
        dist = self.get_prob_dist_x_given_y(class_label)
        prob_x_given_y = pdf(dist, X)
        return prob_x_given_y

    def get_prob_y_given_x(self, X, class_label):
        """
        Get the probability of a flipped class label given the input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        class_label : int
            The class label for which to compute the probability.

        Returns
        -------
        prob_y_given_x: array-like
            The probability of a flipped class label given the input data X.
        """
        pdf_xy = lambda x, k_class: self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x)
        marg_x = self.get_prob_fn_margx()
        x_marg = marg_x(X).sum(axis=0)
        prob_y_given_x = pdf_xy(X, class_label) / x_marg
        return prob_y_given_x

    def generate_samples_for_class(self, k_class):
        """
        Generate synthetic samples for a specific class.

        Parameters
        ----------
        k_class : int
            The class label for which to generate samples.

        Returns
        -------
        data: array-like
            A tuple containing the generated features
        labels: array-like
            A list of labels corresponding to the features
        """
        seed = self.random_state.randint(2 ** 32, dtype="uint32")
        mvn = self.get_prob_dist_x_given_y(k_class)
        n_samples = self.samples_per_class[k_class]
        data = mvn.rvs(n_samples, random_state=seed)
        labels = np.zeros(n_samples, np.int32) + k_class
        return data, labels

    def generate_dataset(self):
        """
       Generate the full synthetic dataset.

       Returns
       -------
       X : array-like of shape (n_samples, n_features)
            Feature matrix after applying sampling to create imbalance.
       y : array-like of shape (n_samples,)
            Target vector after applying sampling to create imbalance.
        """
        X = []
        y = []
        for k_class in self.class_labels:
            data, labels = self.generate_samples_for_class(k_class)
            if len(X) == 0:
                X = data
                y = labels
            else:
                X = np.vstack((X, data))
                y = np.append(y, labels)
        return X, y

    def entropy_y(self, y):
        """
        Calculate the entropy of the class distribution in the dataset.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The labels of the dataset.

        Returns
        -------
        entropy_output: float
            The entropy of the class distribution.
        """
        uni, counts = np.unique(y, return_counts=True)
        y_pred = counts / np.sum(counts)
        y_pred = {i: c for i, c in zip(list(uni), y_pred)}
        entropy_output = 0
        for k_class in self.class_labels:
            entropy_output += -self.y_prob[k_class] * np.log2(self.y_prob[k_class])
            self.logger.info(f"{k_class}: {y_pred[k_class]},  {self.y_prob[k_class]}")
        return entropy_output

    def calculate_mi(self):
        """
        Calculate the mutual information (MI) using the probability distribution function using the formulae below.

        .. math::
            I(X;Y) = H(X) - H(X|Y)

        Returns
        -------
        mutual_information : float
            The mutual information of the dataset.
        """
        x_y_prob_list = []
        for k_class in self.class_labels:
            prob_list = -1
            nter = 0
            while prob_list < 0:
                X, y = self.generate_dataset()
                ind = np.where(y == k_class)[0]
                data = X[ind, :]
                x_y_prob = self.get_prob_x_given_y(X=data, class_label=k_class)
                marg_x = self.get_prob_fn_margx()
                p_x_marg = marg_x(data).sum(axis=0)
                a_log_x_prob = (x_y_prob / p_x_marg)
                prob_list = np.nanmean(np.log2(a_log_x_prob))
                nter += 1
                if nter >= 100:
                    break
            if prob_list < 0:
                prob_list = -1 * prob_list
            x_y_prob_list.append(prob_list * self.y_prob[k_class])
        mi = np.nansum(x_y_prob_list)
        return mi

    def bayes_predictor_mi(self):
        """
        Calculate the mutual information (MI) using the probability distribution function using the formulae below.

        .. math::
            I(X;Y) = H(Y) - H(Y|X)

        Returns
        -------
        mutual_information : float
            The mutual information of the dataset.
        """
        X, y = self.generate_dataset()
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for k_class in self.class_labels:
            y_pred[:, k_class] = self.get_prob_y_given_x(X=X, class_label=k_class)
        y_pred[y_pred == 0] = np.finfo(float).eps
        y_pred[y_pred == 1] = 1 - np.finfo(float).eps
        pyx = (y_pred * np.log2(y_pred)).sum(axis=1)
        mi_bp = pyx.mean()
        self.ent_y = self.entropy_y(y)

        mi = mi_bp + self.ent_y
        self.logger.info(f"mi_bp {mi_bp} mi_pp {self.ent_y}")
        mi = np.max([mi, 0.0])
        return mi

    def bayes_predictor_pc_softmax_mi(self):
        """
        Calculate the mutual information (MI) using class probabilities derived from the PDF of a class label given the
        input data X, applying both the Softmax and PC-Softmax functions.

        .. math::

            I(X;Y) = H(Y) - H(Y|X)

        Softmax Function:

        .. math::

            S(z_k) = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}}

        where:
            - \( z_k \) is the logit or raw score for class \( k \).
            - \( K \) is the total number of classes.

        PC-Softmax (Probability-Corrected Softmax) Function:

        .. math::

            S_{pc}(z_k) = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j} \\cdot p_j}

        where:
            - \( z_k \) is the logit or raw score for class \( k \).
            - \( p_j \) is the prior probability of class \( j \), calculated as \( p_j = \frac{\text{counts}_j}{\text{total samples}} \).

        Returns
        -------
        softmax_emi : float
            Estimated softmax mutual information.

        pc_softmax_emi : float
            Estimated PC-softmax mutual information.
        """
        X, y = self.generate_dataset()
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for k_class in self.class_labels:
            y_pred[:, k_class] = self.get_prob_y_given_x(X=X, class_label=k_class)

        y_pred[y_pred == 0] = np.finfo(float).eps
        y_pred[y_pred == 1] = 1 - np.finfo(float).eps
        classes, counts = np.unique(y, return_counts=True)
        pys = counts / np.sum(counts)

        normal_softmaxes = softmax(y_pred)
        pc_softmax_mis = []
        softmax_mis = []
        x_exp = np.exp(y_pred)
        weighted_x_exp = x_exp * pys
        # weighted_x_exp = x_exp
        x_exp_sum = np.sum(weighted_x_exp, axis=1, keepdims=True)
        pc_softmaxies = x_exp / x_exp_sum
        for i, y_t in enumerate(y):
            mi = np.log2(pc_softmaxies[i, int(y_t)])
            # print("########################################################################")
            # print(f"y_t {y_t} mi {mi} pc_softmaxies {pc_softmaxies[i]} y_pred {y_pred[i]}")
            pc_softmax_mis.append(mi)

            mi = np.log2(normal_softmaxes[i, int(y_t)]) + np.log2(self.n_classes)
            # print(f"y_t {y_t} mi {mi} normal_softmaxes {normal_softmaxes[i]} y_pred {y_pred[i]} LogM {np.log(self.n_classes)}")
            softmax_mis.append(mi)
        # print(pc_softmax_mis, softmax_mis)
        pc_softmax_emi = np.nanmean(pc_softmax_mis)
        softmax_emi = np.nanmean(softmax_mis)
        return softmax_emi, pc_softmax_emi

    def get_bayes_mi(self, metric_name):
        """
        Get the estimated mutual information based on the specified metric.

        Parameters
        ----------
        metric_name : {`MCMCBayesMI`, `MCMCLogLossBayesMI`, `MCMCPCSoftmaxBayesMI`, `MCMCSoftmaxBayesMI`}, default=`MCMCLogLossBayesMI`
            The name of the metric to use for MI estimation.
            Must be one of:

            - `MCMCLogLossBayesMI`: Estimate mutual information using the log loss of the bayes pedictor.
            - `MCMCBayesMI`: Estimate mutual information using the marginal of inputs and conditionals on inputs using class labels
            - `MCMCPCSoftmaxBayesMI`: Estimate mutual information using the MCMC PC Softmax Bayes method.
            - `MCMCSoftmaxBayesMI`: Estimate mutual information using the MCMC Softmax Bayes method.

        Returns
        -------
        mutual_information: float
            The estimated mutual information based on the selected metric.
        """
        mutual_information = 0
        if metric_name == MCMC_LOG_LOSS:
            mutual_information = self.bayes_predictor_mi()
        if metric_name == MCMC_MI_ESTIMATION:
            mutual_information = self.calculate_mi()
        softmax_emi, pc_softmax_emi = self.bayes_predictor_pc_softmax_mi()
        if metric_name == MCMC_PC_SOFTMAX:
            mutual_information = pc_softmax_emi
        if metric_name == MCMC_SOFTMAX:
            mutual_information = softmax_emi
        mutual_information = np.max([mutual_information, 0.0])
        return mutual_information
