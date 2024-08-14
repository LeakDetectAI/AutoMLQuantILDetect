import torch
import torch.nn.functional as F
from torch import nn


def own_softmax(x, label_proportions, device):
    """
    Custom softmax function that incorporates label proportions to handle imbalanced data.

    This function computes a modified softmax, where the exponentiated logits are weighted by the proportions of each
    class label. This can help in cases where class imbalance is significant, ensuring that the model accounts for the
    distribution of labels during prediction.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor (logits) of shape `(n_samples, n_classes)`.
    label_proportions : list, numpy.ndarray, or torch.Tensor
        The proportions of each class in the dataset. This should be a list or tensor of shape `(n_classes,)`
        representing the proportion of each class in the dataset.
    device : torch.device
        The device on which to perform the computation (e.g., 'cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        The resulting tensor after applying the weighted softmax operation, of shape `(n_samples, n_classes)`.

    Notes
    -----
    This function first exponentiates the logits (`x`) and then multiplies them by the corresponding class proportions
    (`label_proportions`). The resulting tensor is normalized by the sum of the weighted exponentiated logits to produce
    a probability distribution across classes.
    """
    if not isinstance(label_proportions, torch.Tensor):
        label_proportions = torch.tensor(label_proportions).to(device)

    x_exp = torch.exp(x)
    weighted_x_exp = x_exp * label_proportions
    x_exp_sum = torch.sum(weighted_x_exp, 1, keepdim=True)

    return x_exp / x_exp_sum


class ClassNet(nn.Module):
    """
    ClassNet is a fully connected neural network used for classification tasks.

    This class implements a simple feedforward neural network with a configurable number of hidden layers and units.
    It supports a custom softmax function (PC-softmax) for handling imbalanced data.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output classes.
    n_units : int
        Number of units in each hidden layer.
    n_hidden : int
        Number of hidden layers.
    device : torch.device
        Device to run the network on (CPU or GPU).
    is_pc_softmax : bool, optional, default=True
        If True, use the custom PC-softmax function in the final layer.

    Attributes
    ----------
    input : torch.nn.Linear
        Input layer.
    hidden_layers : list of torch.nn.Linear
        Hidden layers.
    output : torch.nn.Linear
        Output layer.
    is_pc_softmax : bool
        Whether to use the PC-softmax function.
    device : torch.device
        Device used for computation.
    """

    def __init__(self, in_dim, out_dim, n_units, n_hidden, device, is_pc_softmax=True):
        super(ClassNet, self).__init__()
        self.input = nn.Linear(in_dim, n_units)
        self.hidden_layers = [nn.Linear(n_units, n_units) for _ in range(n_hidden - 1)]
        self.output = nn.Linear(n_units, out_dim)
        self.is_pc_softmax = is_pc_softmax
        self.device = device

    def forward(self, x_in, label_proportions):
        """
        Forward pass through the network.

        Parameters
        ----------
        x_in : torch.Tensor
            Input tensor.
        label_proportions : list or torch.Tensor
            Proportions of each class in the dataset.

        Returns
        -------
        x_in : torch.Tensor
            Output tensor after applying the network layers.
        """
        x_in = torch.relu(self.input(x_in))
        for i, hidden in enumerate(self.hidden_layers):
            x_in = torch.relu(hidden(x_in))
        x_in = self.output(x_in)
        if label_proportions is not None and self.is_pc_softmax:
            x_in = torch.log(own_softmax(x_in, label_proportions, self.device) + 1e-6)
        else:
            x_in = torch.log(F.softmax(x_in, dim=1) + 1e-6)
        return x_in

    def score(self, x_in, label_proportions):
        """
        Compute class probabilities for the input samples.

        Parameters
        ----------
        x_in : torch.Tensor
            Input tensor.
        label_proportions : list or torch.Tensor
            Proportions of each class in the dataset.

        Returns
        -------
        x_in : torch.Tensor
            Output tensor with class probabilities.
        """
        x_in = torch.relu(self.input(x_in))
        for i, hidden in enumerate(self.hidden_layers):
            x_in = torch.relu(hidden(x_in))
        x_in = self.output(x_in)
        x_in = F.softmax(x_in, dim=1)
        return x_in

class StatNet(nn.Module):
    """
    StatNet is a fully connected neural network used for statistical modeling in MINE (Mutual Information Neural Estimation)
    tasks to estimate mutual information.

    This class implements a simple feedforward neural network with a configurable number of hidden layers and units.
    It is typically used to model the joint distribution of input features and class labels for MI estimation.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    cls_enc : int, optional, default=1
        Number of classes in the one-hot encoded target variable.
    n_units : int, optional, default=100
        Number of units in each hidden layer.
    n_hidden : int, optional, default=1
        Number of hidden layers.
    device : torch.device, optional, default='cpu'
        Device to run the network on (CPU or GPU).

    Attributes
    ----------
    input : torch.nn.Linear
        Input layer.
    hidden_layers : list of torch.nn.Linear
        Hidden layers.
    output : torch.nn.Linear
        Output layer.
    """

    def __init__(self, in_dim, cls_enc=1, n_units=100, n_hidden=1, device='cpu'):
        super(StatNet, self).__init__()
        self.device = device
        self.input = nn.Linear(in_dim + cls_enc, n_units).to(self.device)
        self.hidden_layers = [nn.Linear(n_units, n_units).to(self.device) for _ in range(n_hidden - 1)]
        self.output = nn.Linear(n_units, 1).to(self.device)

    def forward(self, x_in):
        """
        Forward pass through the network.

        Parameters
        ----------
        x_in : torch.Tensor
            Input tensor.

        Returns
        -------
        x_in : torch.Tensor
            Output tensor after applying the network layers.
        """
        x_in = x_in.to(self.device)
        x_in = torch.relu(self.input(x_in))
        for i, hidden in enumerate(self.hidden_layers):
            x_in = torch.relu(hidden(x_in))
        x_in = self.output(x_in)
        return x_in
