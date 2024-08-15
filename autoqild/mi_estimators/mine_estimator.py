"""Mutual Information Neural Estimator (MINE) that uses multiple deep learning architectures to estimate MI for
classification tasks."""
import logging
from itertools import product

import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from autoqild.mi_estimators.mi_base_class import MIEstimatorBase
from .neural_networks_torch import StatNet
from .pytorch_utils import get_optimizer_and_parameters, init, get_mine_loss
from ..utilities import softmax


class MineMIEstimator(MIEstimatorBase):
    """
    MineMIEstimator class implementing the Mutual Information Neural Estimator (MINE) approach to estimate the
    mutual information using an ensemble of deep neural networks.

    This class trains multiple neural networks with varying architectures to estimate the mutual information (MI)
    between input features and class labels. By aggregating predictions across an ensemble of models, the estimator
    achieves a more stable and accurate MI estimate. The model is particularly useful when there is a need for
    robust MI estimates in high-dimensional data with complex relationships.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification data samples.

    n_features : int
        Number of features or dimensionality of the inputs of the classification data samples.

    loss_function : {`donsker_varadhan`, `donsker_varadhan_softplus`, `fdivergence`}, default=`donsker_varadhan_softplus`
        The divergence metric to use for the MINE loss.
        Options include:

        - `donsker_varadhan`: Donsker-Varadhan representation of KL divergence.
        - `donsker_varadhan_softplus`: Softplus version of the Donsker-Varadhan representation.
        - `fdivergence`: f-divergence representation of mutual information.

    optimizer_str : {`RMSprop`, `sgd`, `adam`, `AdamW`, `Adagrad`, `Adamax`, `Adadelta`}, default=`adam`
        Optimizer type to use for training the neural network.
        Must be one of:

        - `RMSprop`: Root Mean Square Propagation, an adaptive learning rate method.
        - `sgd`: Stochastic Gradient Descent, a simple and widely-used optimizer.
        - `adam`: Adaptive Moment Estimation, combining momentum and RMSProp for better convergence.
        - `AdamW`: Adam with weight decay, an improved variant of Adam with better regularization.
        - `Adagrad`: Adaptive Gradient Algorithm, adjusting the learning rate based on feature frequency.
        - `Adamax`: Variant of Adam based on infinity norm, more robust with sparse gradients.
        - `Adadelta`: An extension of Adagrad that seeks to reduce its aggressive learning rate decay.

    learning_rate : float, optional, default=1e-4
        Learning rate for the optimizer.

    reg_strength : float, optional, default=0
        Regularization strength.

    encode_classes : bool, optional, default=True
        Indicates if the target variable should be one-hot encoded.

    random_state : int, optional, default=42
        Random state for reproducibility.

    Attributes
    ----------
    optimizer_cls : object
        Optimizer class selected based on the `optimizer_str` parameter.

    device : torch.device
        Device on which the model runs (`cuda` or `cpu`).

    models : list
        List to store the trained models for each configuration.

    n_models : int
        Number of models trained.

    label_binarizer : LabelBinarizer
        LabelBinarizer instance for encoding class labels.

    final_loss : float
        The final average loss over all trained models.

    mi_validation_final : float
        The final average mutual information validation score.

    Notes
    -----
    The MineMIEstimator trains multiple models with varying configurations (e.g., different hidden layers and units).
    This ensemble approach allows the estimator to aggregate results from multiple models to produce a more robust
    estimate of mutual information. The method is particularly effective in cases where the relationships between
    features and labels are complex or non-linear, as the aggregation process helps to smooth out inconsistencies
    across individual model predictions.

    Private Methods
    ---------------
    __pytorch_tensor_dataset__:
        Create PyTorch tensor datasets for the input features and target labels.

    Example
    -------
    >>> estimator = MineMIEstimator(n_classes=3, n_features=10)
    >>> estimator.fit(X_train, y_train)
    >>> mi_estimate = estimator.estimate_mi(X_test, y_test)
    >>> print(mi_estimate)
    """

    def __init__(self, n_classes, n_features, loss_function="donsker_varadhan_softplus", optimizer_str="adam",
                 learning_rate=1e-4, reg_strength=0, encode_classes=True, random_state=42):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state)
        self.logger = logging.getLogger(MineMIEstimator.__name__)
        self.optimizer_str = optimizer_str
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.optimizer_cls, self._optimizer_config = get_optimizer_and_parameters(optimizer_str, learning_rate,
                                                                                  reg_strength)
        self.encode_classes = encode_classes
        self.loss_function = loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"device {self.device} cuda {torch.cuda.is_available()} device {torch.cuda.device_count()}")
        self.optimizer = None
        self.dataset_properties = None
        self.label_binarizer = None
        self.final_loss = 0
        self.mi_validation_final = 0
        self.models = []
        self.n_models = 0

    def __pytorch_tensor_dataset__(self, X, y, i=2):
        """
        Create PyTorch tensor datasets for the input features and target labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        i : int, optional, default=2
            Seed increment for reproducibility.

        Returns
        -------
        tensor_xy : torch.Tensor
            Tensor containing the original data and labels.

        tensor_xy_tilde : torch.Tensor
            Tensor containing the permuted data and labels.
        """
        seed = self.random_state.randint(2 ** 31, dtype="uint32") + i
        rs = np.random.RandomState(seed)
        if self.encode_classes:
            y_t = self.label_binarizer.transform(y)
            xy = np.hstack((X, y_t))
            y_s = rs.permutation(y)
            y_t = self.label_binarizer.transform(y_s)
            xy_tilde = np.hstack((X, y_t))
        else:
            xy = np.hstack((X, y[:, None]))
            y_s = rs.permutation(y)
            xy_tilde = np.hstack((X, y_s[:, None]))
        tensor_xy = torch.tensor(xy, dtype=torch.float32).to(self.device)  # transform to torch tensor
        tensor_xy_tilde = torch.tensor(xy_tilde, dtype=torch.float32).to(self.device)
        return tensor_xy, tensor_xy_tilde

    def fit(self, X, y, epochs=100000, verbose=0, **kwd):
        """
        Fit the ensemble of MINE neural networks with different architectures and estimate mutual information.

        The ensemble method trains multiple neural networks with varying configurations (e.g., number of hidden layers
        and units) and aggregates their mutual information estimates. This aggregation produces a more stable and
        robust estimate by reducing the variance associated with individual models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        epochs : int, optional, default=100000
            Number of training epochs.

        verbose : int, optional, default=0
            Verbosity level.

        **kwd : dict, optional
            Additional keyword arguments.

        Returns
        -------
        self : MineMIEstimator
            Fitted estimator.
        """
        MON_FREQ = epochs // 10
        MON_ITER = epochs // 50
        if self.encode_classes:
            y_t = LabelBinarizer().fit_transform(y)
            cls_enc = y_t.shape[-1]
        else:
            cls_enc = 1
        self.label_binarizer = LabelBinarizer().fit(y)
        n_hidden_layers = [1, 3, 5]
        n_hidden_units = [8, 64, 128, 256]
        self.final_loss = 0
        self.mi_validation_final = 0
        self.models = []
        self.n_models = 0
        for n_unit, n_hidden in product(n_hidden_layers, n_hidden_units):
            stat_net = StatNet(in_dim=self.n_features, cls_enc=cls_enc, n_hidden=n_hidden, n_units=n_unit)
            stat_net.apply(init)
            stat_net.to(self.device)
            optimizer = self.optimizer_cls(stat_net.parameters(), **self._optimizer_config)
            all_estimates = []
            sum_loss = 0
            for iter_ in tqdm(range(epochs), total=epochs, desc="iteration"):
                stat_net.zero_grad()
                xy, xy_tilde = self.__pytorch_tensor_dataset__(X, y, i=iter_)
                preds_xy = stat_net(xy)
                preds_xy_tilde = stat_net(xy_tilde)
                train_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                loss = train_div.mul_(-1.)
                loss.backward()
                optimizer.step()
                sum_loss += loss
                if (iter_ % MON_FREQ == 0) or (iter_ + 1 == epochs):
                    with torch.no_grad():
                        mi_hats = []
                        for _ in range(MON_ITER):
                            xy, xy_tilde = self.__pytorch_tensor_dataset__(X, y, i=iter_)
                            preds_xy = stat_net(xy)
                            preds_xy_tilde = stat_net(xy_tilde)
                            eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                            mi_hats.append(eval_div.cpu().numpy())
                        mi_hat = np.mean(mi_hats)
                        if verbose:
                            print(f"iter: {iter_}, MI hat: {mi_hat} Loss: {loss.detach().numpy()[0]}")
                        self.logger.info(f"iter: {iter_}, MI hat: {mi_hat} Loss: {loss.detach().numpy()[0]}")
                        all_estimates.append(mi_hat)
            final_loss = sum_loss.detach().numpy()[0]
            mis = np.array(all_estimates)
            n = int(len(all_estimates) / 3)
            mi_val = np.nanmean(mis[np.argpartition(mis, -n)[-n:]])
            self.models.append(stat_net)
            self.final_loss += final_loss
            self.mi_validation_final += mi_val
            self.logger.info(f"Fit Loss {final_loss} MI Val: {mi_val} for n_hidden {n_hidden} n_unit {n_unit}")
        self.n_models = len(self.models)
        self.final_loss = self.final_loss / self.n_models
        self.mi_validation_final = self.mi_validation_final / self.n_models

        return self

    def predict(self, X, verbose=0):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        scores = self.predict_proba(X=X, verbose=verbose)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        """
        Compute the score of the ensemble MINE model.

        The score is based on the mutual information estimated by aggregating results from multiple trained models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            The score of the model based on the final estimated mutual information.
        """
        mutual_information = self.mi_validation_final
        self.logger.info(f"Loss {self.final_loss} MI Val: {self.mi_validation_final}")
        if np.isnan(mutual_information) or np.isinf(mutual_information):
            mutual_information = 0.0
        return mutual_information

    def predict_proba(self, X, verbose=0):
        """
        Predict class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        p_pred : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        p_pred = self.decision_function(X=X, verbose=verbose)
        p_pred = softmax(p_pred)
        return p_pred

    def decision_function(self, X, verbose=0):
        """
        Predict confidence scores for samples.

        This method aggregates the confidence scores across all models in the ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        final_scores : array-like of shape (n_samples, n_classes)
            Predicted confidence scores.
        """
        scores = None
        final_scores = None
        for model in self.models:
            for n_class in range(self.n_classes):
                y = np.zeros(X.shape[0]) + n_class
                xy, xy_tilde = self.__pytorch_tensor_dataset__(X, y, i=0)
                score = model(xy).detach().numpy()
                if scores is None:
                    scores = score
                else:
                    scores = np.hstack((scores, score))
            final_scores += scores
        final_scores = final_scores / self.n_models
        return final_scores

    def estimate_mi(self, X, y, verbose=0, MON_ITER=1000, **kwargs):
        """
        Estimate mutual information by taking a mean of estimates obtained from multiple MINE learned models with
        different architectures.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        verbose : int, optional, default=0
            Verbosity level.

        MON_ITER : int, optional, default=1000
            Number of iterations for estimating MI.

        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        mi_estimated : float
            Estimated mutual information.
        """
        final_mis = []
        for model in self.models:
            mi_hats = []
            for iter_ in range(MON_ITER):
                xy, xy_tilde = self.__pytorch_tensor_dataset__(X, y, i=iter_)
                preds_xy = model(xy)
                preds_xy_tilde = model(xy_tilde)
                eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                mi_hat = eval_div.detach().numpy().flatten()[0]
                if verbose:
                    print(f"iter: {iter_}, MI hat: {mi_hat}")
                mi_hats.append(mi_hat)
            mi_hats = np.array(mi_hats)
            n = int(MON_ITER / 2)
            mi_hats = mi_hats[np.argpartition(mi_hats, -n)[-n:]]
            mi_estimated = np.nanmean(mi_hats)
            if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                self.logger.error("Setting MI to 0")
                mi_estimated = 0
            self.logger.info(f"Estimated MIs: {mi_hats[-10:]} Mean {mi_estimated}")
            mi_estimated = np.max([mi_estimated, 0.0])
            final_mis.append(mi_estimated)
        mi_estimated = np.nanmedian(final_mis)
        mi_estimated = np.nanmax([mi_estimated, 0.0])
        return mi_estimated