import logging
from abc import ABCMeta

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from sklearn.utils import check_random_state, shuffle

MCMC_MI_ESTIMATION = 'MCMCBayesMI'
MCMC_LOG_LOSS = 'MCMCLogLossBayesMI'
MCMC_PC_SOFTMAX = 'MCMCPCSoftmaxBayesMI'
MCMC_SOFTMAX = 'MCMCSoftmaxBayesMI'

def pdf(dist, x):
    return np.exp(dist.logpdf(x))


def logsumexp(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


def softmax(x, axis=1):
    """
    Take softmax for the given numpy array.
    :param axis: The axis around which the softmax is applied
    :param x: array-like, shape (n_samples, ...)
    :return: softmax taken around the given axis
    """
    lse = logsumexp(x, axis=axis)
    return np.exp(x - lse)

class SyntheticDatasetGenerator(metaclass=ABCMeta):
    def __init__(self, n_classes=2, n_features=2, samples_per_class=500, flip_y=0.1, random_state=42, fold_id=0,
                 imbalance=0.0, gen_type='single', **kwargs):
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
        self.flip_y_prob = {}
        self.flip_y = flip_y
        self.generate_cov_means()
        self.logger = logging.getLogger(SyntheticDatasetGenerator.__name__)

    def generate_cov_means(self):
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
            mean = np.ones(self.n_features) + (k_class * 1.5)
            self.means[k_class] = mean
            self.covariances[k_class] = cov
            self.seeds[k_class] = seed
            self.y_prob[k_class] = self.samples_per_class[k_class] / self.n_instances
            self.flip_y_prob[k_class] = (self.y_prob[k_class] * (1 - self.flip_y) + self.flip_y / self.n_classes)
            self.flip_y_prob[k_class] = np.around(self.flip_y_prob[k_class], 4)
        # print(self.y_prob)
        # print(self.flip_y_prob)

    def get_prob_dist_x_given_y(self, k_class):
        return multivariate_normal(mean=self.means[k_class], cov=self.covariances[k_class],
                                   seed=self.seeds[k_class])

    def get_prob_fn_margx(self):
        marg_x = lambda x: np.array([self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x)
                                     for k_class in self.class_labels])
        return marg_x

    def get_prob_x_given_y(self, X, class_label):
        dist = self.get_prob_dist_x_given_y(class_label)
        prob_x_given_y = pdf(dist, X)
        return prob_x_given_y

    def get_prob_y_given_x(self, X, class_label):
        pdf_xy = lambda x, k_class: self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x)
        marg_x = self.get_prob_fn_margx()
        x_marg = marg_x(X).sum(axis=0)
        prob_y_given_x = pdf_xy(X, class_label) / x_marg
        return prob_y_given_x

    def get_prob_flip_y_given_x(self, X, class_label):
        first_term = (1 - self.flip_y) * self.get_prob_y_given_x(X, class_label)
        # second_term = (self.flip_y / self.n_classes)
        second_term = (self.flip_y * self.y_prob[class_label])
        prob_y_given_x = first_term + second_term
        return prob_y_given_x

    def get_prob_x_given_flip_y(self, X, class_label):
        prob_flip_y_given_x = self.get_prob_flip_y_given_x(X, class_label)
        marg_x = self.get_prob_fn_margx()
        x_marg = marg_x(X).sum(axis=0)
        prob_x_given_flip_y = (prob_flip_y_given_x * x_marg)
        return prob_x_given_flip_y

    def generate_samples_for_class(self, k_class):
        seed = self.random_state.randint(2 ** 32, dtype="uint32")
        mvn = self.get_prob_dist_x_given_y(k_class)
        n_samples = self.samples_per_class[k_class]
        data = mvn.rvs(n_samples, random_state=seed)
        labels = np.zeros(n_samples, dtype='int64') + k_class
        return data, labels

    def generate_dataset(self):
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
        if self.flip_y > 0:
            if len(np.unique(list(self.samples_per_class.values()))) == 1:
                self.logger.info("################# Balanced Dataset #################")
                indices = []
                for k_class in self.class_labels:
                    flip_samples = int(self.flip_y * self.samples_per_class[k_class])
                    ind0 = list(self.random_state.choice(np.where(y == k_class)[0], flip_samples, replace=False))
                    indices.extend(ind0)
                uni, counts = np.unique(y, return_counts=True)
                d = {i: c for i, c in zip(uni, counts)}
                self.logger.info(f"Original {d}")
                self.logger.info(f"Probs {self.y_prob}")
                y[indices] = shuffle(y[indices], random_state=self.random_state)
                uni, counts = np.unique(y, return_counts=True)
                d = {i: c for i, c in zip(list(uni), counts)}
                d_o = {i: c for i, c in zip(list(uni), counts / np.sum(counts))}
                self.logger.info(f"After Flipping {d}")
                self.logger.info(f"Flipping Ratio {d_o}")
            else:
                self.logger.info("################# Imbalanced Dataset #################")
                uni, counts = np.unique(y, return_counts=True)
                d = {i: c for i, c in zip(uni, counts)}
                self.logger.info(f"Original {d}")
                self.logger.info(f"Probs {self.y_prob}")
                choices = []
                indices = []
                for i, y_i in enumerate(y):
                    choice = self.random_state.choice(2, 1, p=[1 - self.flip_y, self.flip_y])
                    choices.append(choice)
                    if choice == 1:
                        indices.append(i)
                        # y[i] = self.random_state.choice(self.n_classes, 1)
                p = np.array([self.samples_per_class[int(k)] / self.n_instances for k in range(self.n_classes)])
                y_old = np.copy(y)
                y[indices] = self.random_state.choice(self.n_classes, size=len(indices), p=p)
                # y[indices] = self.random_state.randint(self.n_classes, size=len(indices))
                self.logger.info(f"Actual Flip {self.flip_y} Flips {np.mean(y_old != y)}")
                self.logger.info(f"Chosen flips {np.mean(choices)}")
                uni, counts = np.unique(y, return_counts=True)
                d = {i: c for i, c in zip(list(uni), counts)}
                d_o = {i: c for i, c in zip(list(uni), counts / np.sum(counts))}
                self.logger.info(f"After Flipping {d}")
                self.logger.info(f"Flipping Ratio {d_o}")
                self.logger.info(f"Y_prob {self.y_prob}")
                self.logger.info(f"Probs after flipping {self.flip_y_prob}")
        return X, y

    def entropy_y(self, y):
        uni, counts = np.unique(y, return_counts=True)
        y_pred = counts / np.sum(counts)
        y_pred = {i: c for i, c in zip(list(uni), y_pred)}
        mi_pp = 0
        for k_class in self.class_labels:
            mi_pp += -self.y_prob[k_class] * np.log2(self.y_prob[k_class])
            self.logger.info(f"{k_class}: {y_pred[k_class]},  {self.y_prob[k_class]}")
        return mi_pp

    def calculate_mi(self):
        x_y_prob_list = []
        for k_class in self.class_labels:
            prob_list = -1
            nter = 0
            while prob_list < 0:
                X, y = self.generate_dataset()
                ind = np.where(y == k_class)[0]
                data = X[ind, :]
                if self.flip_y == 0.0:
                    x_y_prob = self.get_prob_x_given_y(X=data, class_label=k_class)
                    marg_x = self.get_prob_fn_margx()
                    p_x_marg = marg_x(data).sum(axis=0)
                    a_log_x_prob = (x_y_prob / p_x_marg)
                else:
                    x_y_prob = self.get_prob_flip_y_given_x(X=data, class_label=k_class)
                    a_log_x_prob = (x_y_prob / self.y_prob[k_class])
                    # print(x_y_prob)
                prob_list = np.nanmean(np.log2(a_log_x_prob))
                # print(prob_list)
                nter = nter + 1
                if nter >= 100:
                    break
            if prob_list < 0:
                prob_list = -1 * prob_list
            if self.flip_y == 0.0:
                x_y_prob_list.append(prob_list * self.y_prob[k_class])
            else:
                x_y_prob_list.append(prob_list * self.y_prob[k_class])
        mi = np.nansum(x_y_prob_list)
        return mi

    def bayes_predictor_mi(self):
        X, y = self.generate_dataset()
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for k_class in self.class_labels:
            if self.flip_y == 0.0:
                y_pred[:, k_class] = self.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                y_pred[:, k_class] = self.get_prob_flip_y_given_x(X=X, class_label=k_class)
        y_pred[y_pred == 0] = np.finfo(float).eps
        y_pred[y_pred == 1] = 1 - np.finfo(float).eps
        self.logger.info(f"Sum {y_pred.sum(axis=1)}")
        self.logger.info(f"Sum {y_pred.sum()}, n_instances {self.n_instances}")
        # self.logger.info(f"Y_pred {np.around(y_pred[0:3, :], 4)}")
        pyx = (y_pred * np.log2(y_pred)).sum(axis=1)
        mi_bp = pyx.mean()
        self.ent_y = self.entropy_y(y)
        mi = mi_bp + self.ent_y
        self.logger.info(f"mi_bp {mi_bp} mi_pp {self.ent_y}")
        mi = np.max([mi, 0.0])
        return mi

    def bayes_predictor_pc_softmax_mi(self):
        X, y = self.generate_dataset()
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for k_class in self.class_labels:
            if self.flip_y == 0.0:
                y_pred[:, k_class] = self.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                y_pred[:, k_class] = self.get_prob_flip_y_given_x(X=X, class_label=k_class)

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
        pc_softmaxes = x_exp / x_exp_sum
        for i, y_t in enumerate(y):
            mi = np.log2(pc_softmaxes[i, int(y_t)])
            # print("########################################################################")
            # print(f"y_t {y_t} mi {mi} pc_softmaxes {pc_softmaxes[i]} y_pred {y_pred[i]}")
            pc_softmax_mis.append(mi)

            mi = np.log2(normal_softmaxes[i, int(y_t)]) + np.log2(self.n_classes)
            # print(f"y_t {y_t} mi {mi} normal_softmaxes {normal_softmaxes[i]} y_pred {y_pred[i]} LogM {np.log(self.n_classes)}")
            softmax_mis.append(mi)
        # print(pc_softmax_mis, softmax_mis)
        pc_softmax_emi = np.nanmean(pc_softmax_mis)
        softmax_emi = np.nanmean(softmax_mis)
        return softmax_emi, pc_softmax_emi

    def get_bayes_mi(self, metric_name):
        mi = 0
        if metric_name == MCMC_LOG_LOSS:
            mi = self.bayes_predictor_mi()
        if metric_name == MCMC_MI_ESTIMATION:
            mi = self.calculate_mi()
        softmax_emi, pc_softmax_emi = self.bayes_predictor_pc_softmax_mi()
        if metric_name == MCMC_PC_SOFTMAX:
            mi = pc_softmax_emi
        if metric_name == MCMC_SOFTMAX:
            mi = softmax_emi
        mi = np.max([mi, 0.0])
        return mi
