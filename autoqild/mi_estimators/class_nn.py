import torch
import torch.nn.functional as F
from torch import nn


def own_softmax(x, label_proportions, device):
    if not isinstance(label_proportions, torch.Tensor):
        label_proportions = torch.tensor(label_proportions).to(device)
    x_exp = torch.exp(x)
    weighted_x_exp = x_exp * label_proportions
    # weighted_x_exp = x_exp
    x_exp_sum = torch.sum(weighted_x_exp, 1, keepdim=True)

    return x_exp / x_exp_sum


class ClassNet(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_hidden, device, is_pc_softmax=True):
        super(ClassNet, self).__init__()
        self.input = nn.Linear(in_dim, n_units)
        self.hidden_layers = [nn.Linear(n_units, n_units) for x in range(n_hidden - 1)]
        self.output = nn.Linear(n_units, out_dim)
        self.is_pc_softmax = is_pc_softmax
        self.device = device

    def forward(self, x_in, label_proportions):
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
        x_in = torch.relu(self.input(x_in))
        for i, hidden in enumerate(self.hidden_layers):
            x_in = torch.relu(hidden(x_in))
        x_in = self.output(x_in)
        x_in = F.softmax(x_in, dim=1)
        return x_in


class StatNet(nn.Module):
    def __init__(self, in_dim, cls_enc=1, n_units=100, n_hidden=1, device='cpu'):
        super(StatNet, self).__init__()
        self.device = device
        self.input = nn.Linear(in_dim + cls_enc, n_units).to(self.device)
        self.hidden_layers = [nn.Linear(n_units, n_units).to(self.device) for x in range(n_hidden - 1)]
        self.output = nn.Linear(n_units, 1).to(self.device)

    def forward(self, x_in):
        x_in = x_in.to(self.device)  # Move input tensor to the same device as the linear layer
        x_in = torch.relu(self.input(x_in))
        for i, hidden in enumerate(self.hidden_layers):
            x_in = torch.relu(hidden(x_in))
        x_in = self.output(x_in)
        return x_in
