import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, RMSprop, SGD, Adagrad, Adamax, AdamW, Adadelta

optimizers = {'RMSprop': RMSprop, 'sgd': SGD, 'adam': Adam, 'AdamW': AdamW, 'Adagrad': Adagrad, 'Adamax': Adamax,
              'Adadelta': Adadelta}
optimizer_parameters = {'RMSprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-08, 'weight_decay': 0, 'momentum': 0,
                                    'centered': False},
                        'sgd': {'lr': 0.001, 'momentum': 0.7, 'weight_decay': 0},
                        'adam': {'lr': 1e-4, 'betas': (0.5, 0.999), 'weight_decay': 0, 'amsgrad': False},
                        'AdamW': {'lr': 1e-4, 'betas': (0.5, 0.999), 'eps': 1e-08, 'weight_decay': 0.01,
                                  'amsgrad': False},
                        'Adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0, 'initial_accumulator_value': 0,
                                    'eps': 1e-10},
                        'Adamax': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0},
                        'Adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-06, 'weight_decay': 0}}


def get_optimizer_and_parameters(optimizer_str, learning_rate, reg_strength):
    """
    Get the optimizer and its configuration parameters based on the specified optimizer string.

    Parameters
    ----------
    optimizer_str : {'RMSprop', 'sgd', 'adam', 'AdamW', 'Adagrad', 'Adamax', 'Adadelta'}, default='adam'
        Optimizer type to use for training the neural network.
        Must be one of:

        - 'RMSprop': Root Mean Square Propagation, an adaptive learning rate method.
        - 'sgd': Stochastic Gradient Descent, a simple and widely-used optimizer.
        - 'adam': Adaptive Moment Estimation, combining momentum and RMSProp for better convergence.
        - 'AdamW': Adam with weight decay, an improved variant of Adam with better regularization.
        - 'Adagrad': Adaptive Gradient Algorithm, adjusting the learning rate based on feature frequency.
        - 'Adamax': Variant of Adam based on infinity norm, more robust with sparse gradients.
        - 'Adadelta': An extension of Adagrad that seeks to reduce its aggressive learning rate decay.

    learning_rate : float
        The learning rate for the optimizer.

    reg_strength : float
        The regularization strength (weight decay) for the optimizer.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer class.

    optimizer_config : dict
        The configuration parameters for the optimizer.

    Raises
    ------
    ValueError
        If the specified optimizer string is not recognized.
    """
    optimizer = optimizers.get(optimizer_str, 'adam')
    optimizer_config = optimizer_parameters.get(optimizer_str, 'adam')
    optimizer_config['lr'] = learning_rate
    optimizer_config['weight_decay'] = reg_strength

    return optimizer, optimizer_config


def init(m):
    """
    Initialize the weights and biases of a neural network layer.

    Parameters
    ----------
    m : torch.nn.Module
        The neural network layer to initialize.

    Notes
    -----
    This function initializes the weights of a linear layer using orthogonal initialization and sets the biases to zero.
    """
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.)


def log_mean_exp(inputs, dim=None, keepdim=False):
    """
    Compute the log of the mean of the exponentials of input elements.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor.

    dim : int or tuple of ints, optional
        The dimension or dimensions to reduce. If None, reduces all dimensions.

    keepdim : bool, optional
        Whether the output tensor has dim retained or not.

    Returns
    -------
    outputs : torch.Tensor
        The logarithm of the mean of the exponentials of the input tensor.
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().mean(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def get_mine_loss(preds_xy, preds_xy_tilde, metric):
    """
    Calculate the MINE loss based on the specified metric.

    Parameters
    ----------
    preds_xy : torch.Tensor
        Predictions for the joint distribution samples.

    preds_xy_tilde : torch.Tensor
        Predictions for the product of marginals distribution samples.

    metric : {'donsker_varadhan', 'donsker_varadhan_softplus', 'fdivergence'}
        The divergence metric to use for the MINE loss. Options include:

        - 'donsker_varadhan': Donsker-Varadhan representation of KL divergence.
        - 'donsker_varadhan_softplus': Softplus version of the Donsker-Varadhan representation.
        - 'fdivergence': f-divergence representation of mutual information.

    Returns
    -------
    loss : torch.Tensor
        Calculated MINE loss based on the specified metric.

    Raises
    ------
    ValueError
        If the specified metric is not recognized.
    """
    SMALL = 1e-8
    if metric == 'donsker_varadhan':
        loss = preds_xy.mean(dim=0) - log_mean_exp(preds_xy_tilde, dim=0)
        loss = loss * torch.log2(torch.exp(torch.tensor(1.0)))
        return loss
    elif metric == 'donsker_varadhan_softplus':
        loss = torch.log(F.softplus(preds_xy) + SMALL).mean(dim=0) - torch.log(
            F.softplus(preds_xy_tilde).mean(dim=0) + SMALL)
        loss = loss * torch.log2(torch.exp(torch.tensor(1.0)))
        return loss
    elif metric == 'fdivergence':
        loss = preds_xy.mean(dim=0) - torch.exp(preds_xy_tilde - 1).mean(dim=0)
        loss = loss * torch.log2(torch.exp(torch.tensor(1.0)))
        return loss
    else:
        err_msg = f'unrecognized metric {metric}'
        raise ValueError(err_msg)
