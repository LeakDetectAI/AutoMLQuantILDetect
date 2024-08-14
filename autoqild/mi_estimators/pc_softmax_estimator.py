import logging
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from autoqild.mi_estimators.mi_base_class import MIEstimatorBase
from .neural_networks_torch import ClassNet, own_softmax
from .pytorch_utils import get_optimizer_and_parameters, init


class PCSoftmaxMIEstimator(MIEstimatorBase):
    """
    PCSoftmaxMIEstimator estimates Mutual Information (MI) using a neural network trained with a modified softmax function.

    This class uses a neural network to estimate the MI between input features and class labels. The neural network is trained using a custom softmax function that accounts for label proportions, which can help in handling imbalanced data.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification task.
    n_features : int
        Number of features or dimensionality of the input data.
    n_hidden : int, optional, default=10
        Number of hidden layers in the neural network.
    n_units : int, optional, default=100
        Number of units in each hidden layer.
    loss_function : torch.nn.Module, optional, default=torch.nn.NLLLoss()
        Loss function to be used during training.
    optimizer_str : str, optional, default='adam'
        Optimizer to use for training. Options include 'adam', 'sgd', 'RMSprop', etc.
    learning_rate : float, optional, default=0.001
        Learning rate for the optimizer.
    reg_strength : float, optional, default=0.001
        Regularization strength for the optimizer.
    is_pc_softmax : bool, optional, default=False
        If True, use the custom softmax function that accounts for label proportions.
    random_state : int, optional, default=42
        Seed for random number generation to ensure reproducibility.

    Attributes
    ----------
    logger : logging.Logger
        Logger for logging messages and errors.
    optimizer : torch.optim.Optimizer
        Optimizer used for training the neural network.
    class_net : ClassNet
        Instance of the neural network used for classification.
    dataset_properties : list
        Proportions of each class in the dataset.
    final_loss : float
        Final loss value after training.
    mi_val : float
        Estimated mutual information after training.
    device : torch.device
        Device used for computation (CPU or GPU).
    """

    def __init__(self, n_classes, n_features, n_hidden=10, n_units=100, loss_function=nn.NLLLoss(),
                 optimizer_str='adam', learning_rate=0.001, reg_strength=0.001, is_pc_softmax=False, random_state=42):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state)
        self.logger = logging.getLogger(PCSoftmaxMIEstimator.__name__)
        self.optimizer_str = optimizer_str
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.optimizer_cls, self._optimizer_config = get_optimizer_and_parameters(optimizer_str, learning_rate,
                                                                                  reg_strength)
        self.is_pc_softmax = is_pc_softmax
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.loss_function = loss_function
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.class_net = None
        self.dataset_properties = None
        self.final_loss = 0
        self.mi_val = 0

    def pytorch_tensor_dataset(self, X, y, batch_size=32):
        """
        Create a PyTorch dataset and data loader from the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        batch_size : int, optional, default=32
            Number of samples per batch.

        Returns
        -------
        dataset_prop : list
            Proportions of each class in the dataset.
        tra_dataloader : torch.utils.data.DataLoader
            DataLoader for the training data.
        """
        y_l, counts = np.unique(y, return_counts=True)
        total = len(y)
        dataset_prop = [x / total for x in counts]
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)  # transform to torch tensor
        tensor_y = torch.tensor(y, dtype=torch.int64).to(self.device)
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your dataset
        tra_dataloader = DataLoader(my_dataset, num_workers=1, batch_size=batch_size, shuffle=True, drop_last=False,
                                    pin_memory=True)
        return dataset_prop, tra_dataloader

    def fit(self, X, y, epochs=50, verbose=0, **kwd):
        """
        Fit the neural network to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        epochs : int, optional, default=50
            Number of training epochs.
        verbose : int, optional, default=0
            Verbosity level.
        **kwd : dict, optional
            Additional keyword arguments.

        Returns
        -------
        self : PCSoftmaxMIEstimator
            Fitted estimator.
        """
        self.class_net = ClassNet(in_dim=self.n_features, out_dim=self.n_classes, n_hidden=self.n_hidden,
                                  n_units=self.n_units, device=self.device, is_pc_softmax=self.is_pc_softmax)
        self.class_net.apply(init)
        self.class_net.to(self.device)
        self.optimizer = self.optimizer_cls(self.class_net.parameters(), **self._optimizer_config)

        dataset_prop, tra_dataloader = self.pytorch_tensor_dataset(X, y)
        self.dataset_properties = dataset_prop
        self.final_loss = 0
        for epoch in range(1, epochs + 1):
            correct = 0
            running_loss = 0.0
            sum_loss = 0
            for ite_idx, (tensor_x, tensor_y) in enumerate(tra_dataloader):
                tensor_x = tensor_x.to(self.device)
                tensor_y = tensor_y.to(self.device).squeeze()
                preds_ = self.class_net(tensor_x, dataset_prop)
                loss = self.loss_function(preds_, tensor_y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss
                running_loss += loss.item()
            self.final_loss += float(loss.detach().numpy())
            if verbose and epoch % 10 == 0:
                _, predicted = torch.max(preds_, 1)
                correct += (predicted == tensor_y).sum().item()
                accuracy = 100 * correct / tensor_y.size(0)
                print(f'For Epoch: {epoch} Running loss: {running_loss} Accuracy: {accuracy} %')
                self.logger.error(f'For Epoch: {epoch} Running loss: {running_loss} Accuracy: {accuracy} %')
        self.mi_val = self.estimate_mi(X, y, verbose=0)
        self.logger.info(f"Fit Loss {self.final_loss} MI Val: {self.mi_val}")
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
        predicted : array-like of shape (n_samples,)
            Predicted class labels.
        """
        y = np.random.choice(self.n_classes, X.shape[0])
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            a_label = a_label.to(self.device).squeeze()
            test_ = self.class_net(a_data, dataset_prop)
            _, predicted = torch.max(test_, 1)
        return predicted.detach().numpy()

    def score(self, X, y, sample_weight=None, verbose=0):
        """
        Compute the score of the neural network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        score : float
            Negative loss of the model on the validation data.
        """
        y_pred = self.predict(X, verbose=0)
        acc = np.mean(y == y_pred)
        if np.isnan(self.final_loss) or np.isinf(self.final_loss):
            acc = 0.0
        s_pred = self.predict_proba(X, verbose=0)
        pyx = ((s_pred * np.log2(s_pred)).sum(axis=1)).mean()
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        val_loss = 0
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            preds_ = self.class_net(a_data, dataset_prop)
            a_label = a_label.to(self.device).squeeze()
            loss = self.loss_function(preds_, a_label)
            val_loss += loss
        self.logger.info(f"Loss {self.final_loss} Accuracy {acc} pyx {pyx} MI {self.mi_val} Val loss {val_loss}")
        return -val_loss

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
        y = np.random.choice(self.n_classes, X.shape[0])
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            test_ = self.class_net.score(a_data, dataset_prop)
        return test_.detach().numpy()

    def decision_function(self, X, verbose=0):
        """
        Compute the decision function in form of class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        decision : array-like of shape (n_samples, n_classes)
            Decision function values.
        """
        y = np.random.choice(self.n_classes, X.shape[0])
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            test_ = self.class_net.score(a_data, dataset_prop)
        return test_.detach().numpy()

    def estimate_mi(self, X, y, verbose=1, **kwargs):
        """
        Estimate Mutual Information using the trained neural network using PC-sosftmax loss function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,)
            Target labels.
        verbose : int, optional, default=1
            Verbosity level.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        mi_estimated : float
            The estimated mutual information.
        """
        dataset_prop, test_dataset = self.pytorch_tensor_dataset(X, y, batch_size=1)
        softmax_list = []
        for a_data, a_label in test_dataset:
            int_label = a_label.cpu().item()
            a_data = a_data.unsqueeze(0).to(self.device)
            test_ = self.class_net(a_data, dataset_prop)
            if self.is_pc_softmax:
                a_softmax = torch.flatten(own_softmax(test_, dataset_prop, self.device))[int_label]
            else:
                a_softmax = torch.flatten(torch.softmax(test_, dim=-1))[int_label]
            if self.is_pc_softmax:
                softmax_list.append(math.log2(a_softmax.cpu().item()))
            else:
                softmax_list.append(math.log2(a_softmax.cpu().item()) + math.log2(len(dataset_prop)))
        mi_estimated = np.nanmean(softmax_list)
        if np.isnan(mi_estimated) or np.isinf(mi_estimated):
            mi_estimated = 0
        if self.mi_val - mi_estimated > .01:
            mi_estimated = self.mi_val
        mi_estimated = np.max([mi_estimated, 0.0])
        return mi_estimated

