"""Modified MINE estimator that minimizes mean squared error (MSE) to provide
more robust MI estimates."""

import logging

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from autoqild.mi_estimators.mi_base_class import MIEstimatorBase
from .neural_networks_torch import StatNet
from .pytorch_utils import get_optimizer_and_parameters, init, get_mine_loss
from ..utilities import softmax


class MineMIEstimatorMSE(MIEstimatorBase):
    """MineMIEstimatorMSE class implements a Mutual Information Neural
    Estimator (MINE) using Mean Squared Error (MSE) as the primary objective
    function. The class optimizes neural network architecture through
    hyperparameter tuning with the goal of minimizing MSE during estimation.

    This class leverages MINE techniques and is specifically tailored for hyperparameter optimization, enabling
    the selection of the best neural network architecture for estimating mutual information.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification data samples.

    n_features : int
        Number of features or dimensionality of the inputs of the classification data samples.

    n_hidden : int, optional, default=2
        Number of hidden layers in the neural network.

    n_units : int, optional, default=100
        Number of units per hidden layer.

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

    reg_strength : float, optional, default=1e-10
        Regularization strength.

    encode_classes : bool, optional, default=True
        Indicates if the target variable should be one-hot encoded.

    random_state : int, optional, default=42
        Random state for reproducibility.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `MineMIEstimatorMSE` constructor.

    Attributes
    ----------
    optimizer_cls : object
        Optimizer class selected based on the `optimizer_str` parameter.

    device : torch.device
        Device on which the model runs (`cuda` or `cpu`).

    stat_net : StatNet
        Neural network model for estimating mutual information.

    final_loss : float
        The final loss after training the model.

    mi_val : float
        The final estimated mutual information value.

    Notes
    -----
    This class is particularly suited for scenarios involving hyperparameter tuning where the goal is to identify
    the optimal architecture that minimizes MSE during mutual information estimation.

    Example
    -------
    >>> estimator = MineMIEstimatorMSE(n_classes=3, n_features=10)
    >>> estimator.fit(X_train, y_train)
    >>> score = estimator.score(X_test, y_test)
    >>> print(score)
    """

    def __init__(
        self,
        n_classes,
        n_features,
        n_hidden=2,
        n_units=100,
        loss_function="donsker_varadhan_softplus",
        optimizer_str="adam",
        learning_rate=1e-4,
        reg_strength=1e-10,
        encode_classes=True,
        random_state=42,
        **kwargs,
    ):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state)
        self.logger = logging.getLogger(MineMIEstimatorMSE.__name__)
        self.optimizer_str = optimizer_str
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.optimizer_cls, self._optimizer_config = get_optimizer_and_parameters(
            optimizer_str, learning_rate, reg_strength
        )
        self.encode_classes = encode_classes
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.loss_function = loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(
            f"device {self.device} cuda {torch.cuda.is_available()} gpu device {torch.cuda.device_count()}"
        )
        self.optimizer = None
        self.stat_net = None
        self.dataset_properties = None
        self.label_binarizer = None
        self.final_loss = 0
        self.mi_val = 0

    def pytorch_tensor_dataset(self, X, y, batch_size=64, i=2):
        """Create PyTorch tensor datasets for the input features and target
        labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        batch_size : int, optional, default=64
            Size of the batches used for training.

        i : int, optional, default=2
            Seed increment for reproducibility.

        Returns
        -------
        tensor_xy : torch.Tensor
            Tensor containing the original data and labels.

        tensor_xy_tilde : torch.Tensor
            Tensor containing the permuted data and labels.
        """
        seed = self.random_state.randint(2**31, dtype="uint32") + i
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
        indices = rs.choice(xy_tilde.shape[0], size=batch_size)
        xy = xy[indices]
        xy_tilde = xy_tilde[indices]
        tensor_xy = torch.tensor(xy, dtype=torch.float32).to(
            self.device
        )  # transform to torch tensor
        tensor_xy_tilde = torch.tensor(xy_tilde, dtype=torch.float32).to(self.device)
        return tensor_xy, tensor_xy_tilde

    def fit(self, X, y, epochs=100, batch_size=128, verbose=0, **kwd):
        """Fit the MINE model and estimate mutual information.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        epochs : int, optional, default=10000
            Number of training epochs.

        batch_size : int, optional, default=128
            Batch size used for training.

        verbose : int, optional, default=0
            Verbosity level.

        **kwd : dict, optional
            Additional keyword arguments.

        Returns
        -------
        self : MineMIEstimatorMSE
            Fitted estimator.
        """
        MON_FREQ = epochs // 10
        MON_ITER = 10
        if self.encode_classes:
            y_t = LabelBinarizer().fit_transform(y)
            cls_enc = y_t.shape[-1]
        else:
            cls_enc = 1
        self.label_binarizer = LabelBinarizer().fit(y)
        self.stat_net = StatNet(
            in_dim=self.n_features,
            cls_enc=cls_enc,
            n_hidden=self.n_hidden,
            n_units=self.n_units,
            device=self.device,
        )
        self.stat_net.apply(init)
        self.stat_net.to(self.device)
        self.optimizer = self.optimizer_cls(self.stat_net.parameters(), **self._optimizer_config)
        all_estimates = []
        sum_loss = 0
        for iter_ in tqdm(range(epochs), total=epochs, desc="iteration"):
            self.stat_net.zero_grad()
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=batch_size, i=iter_)
            preds_xy = self.stat_net(xy)
            preds_xy_tilde = self.stat_net(xy_tilde)
            train_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
            loss = train_div.mul_(-1.0)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss
            if (iter_ % MON_FREQ == 0) or (iter_ + 1 == epochs):
                with torch.no_grad():
                    mi_hats = []
                    for _ in range(MON_ITER):
                        xy, xy_tilde = self.pytorch_tensor_dataset(
                            X, y, batch_size=batch_size, i=iter_
                        )
                        preds_xy = self.stat_net(xy)
                        preds_xy_tilde = self.stat_net(xy_tilde)
                        eval_div = get_mine_loss(
                            preds_xy, preds_xy_tilde, metric=self.loss_function
                        )
                        mi_hats.append(eval_div.cpu().numpy())
                    mi_hat = np.mean(mi_hats)
                    if verbose:
                        print(
                            f"iter: {iter_}, MI hat: {mi_hat} Loss: {loss.cpu().detach().numpy()[0]}"
                        )
                    self.logger.info(
                        f"iter: {iter_}, MI hat: {mi_hat} Loss: {loss.cpu().detach().numpy()[0]}"
                    )
                    all_estimates.append(mi_hat)
        self.final_loss = sum_loss.cpu().detach().numpy()[0]
        mis = np.array(all_estimates)
        n = int(len(all_estimates) / 3)
        self.mi_val = np.nanmean(mis[np.argpartition(mis, -n)[-n:]])
        torch.no_grad()
        self.logger.info(f"Fit Loss {self.final_loss} MI Val: {self.mi_val}")
        return self

    def predict(self, X, verbose=0):
        """Predict class labels for the input samples.

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
        """Compute the score of the MINE model using the mean squared error
        between the original and permuted samples.

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
            The score of the model using the mean squared error between the original and permuted samples loss.
        """
        torch.no_grad()
        xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0], i=0)
        preds_xy = self.stat_net(xy).cpu().detach().numpy().flatten()
        preds_xy_tilde = self.stat_net(xy_tilde).cpu().detach().numpy().flatten()
        mse = mean_squared_error(preds_xy, preds_xy_tilde)
        self.logger.info(f"MSE {mse}")
        self.logger.info(
            f"Memory allocated {torch.cuda.memory_allocated()} Cached {torch.cuda.memory_cached()}"
        )
        return mse

    def predict_proba(self, X, verbose=0):
        """Predict class probabilities for the input samples.

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
        scores = self.decision_function(X=X, verbose=verbose)
        scores = softmax(scores)
        return scores

    def decision_function(self, X, verbose=0):
        scores = None
        for n_class in range(self.n_classes):
            y = np.zeros(X.shape[0]) + n_class
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0], i=0)
            score = self.stat_net(xy).cpu().detach().numpy()
            if scores is None:
                scores = score
            else:
                scores = np.hstack((scores, score))
        return scores

    def estimate_mi(self, X, y, verbose=0, MON_ITER=100, **kwargs):
        """Estimate mutual information using the MINE model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        verbose : int, optional, default=0
            Verbosity level.

        MON_ITER : int, optional, default=100
            Number of iterations for estimating MI.

        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        mi_estimated : float
            Estimated mutual information.
        """
        mi_hats = []
        for iter_ in range(MON_ITER):
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0], i=iter_)
            preds_xy = self.stat_net(xy)
            preds_xy_tilde = self.stat_net(xy_tilde)
            eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
            mi_hat = eval_div.cpu().detach().numpy().flatten()[0]
            if verbose:
                print(f"iter: {iter_}, MI hat: {mi_hat}")
            mi_hats.append(mi_hat)
        mi_hats = np.array(mi_hats)
        n = int(MON_ITER / 2)
        mi_hats = mi_hats[np.argpartition(mi_hats, -n)[-n:]]
        mi_estimated = np.nanmean(mi_hats)
        if np.isnan(mi_estimated) or np.isinf(mi_estimated):
            self.logger.error(f"Setting MI to 0")
            mi_estimated = 0
        self.logger.info(f"Estimated MIs: {mi_hats[-10:]} Mean {mi_estimated}")
        if self.mi_val - mi_estimated > 0.01:
            mi_estimated = self.mi_val
        mi_estimated = np.max([mi_estimated, 0.0])
        return mi_estimated
