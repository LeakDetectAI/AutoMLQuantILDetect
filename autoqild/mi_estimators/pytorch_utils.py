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
    optimizer = optimizers.get(optimizer_str, 'adam')
    optimizer_config = optimizer_parameters.get(optimizer_str, 'adam')
    optimizer_config['lr'] = learning_rate
    optimizer_config['weight_decay'] = reg_strength

    return optimizer, optimizer_config


def init(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.)


def log_mean_exp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().mean(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def get_mine_loss(preds_xy, preds_xy_tilde, metric):
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
