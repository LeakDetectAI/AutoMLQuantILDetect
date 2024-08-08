import logging
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from .mi_base_class import MIEstimatorBase
from .neural_networks_torch import ClassNet, own_softmax
from .pytorch_utils import get_optimizer_and_parameters, init


class PCSoftmaxMIEstimator(MIEstimatorBase):
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
        self.logger.info(f"device {self.device} cuda {torch.cuda.is_available()} device {torch.cuda.device_count()}")
        self.optimizer = None
        self.class_net = None
        self.dataset_properties = None
        self.final_loss = 0
        self.mi_val = 0

    def pytorch_tensor_dataset(self, X, y, batch_size=32):
        y_l, counts = np.unique(y, return_counts=True)
        total = len(y)
        dataset_prop = list([x / total for x in counts])
        tensor_x = torch.tensor(X, dtype=torch.float32)  # transform to torch tensor
        tensor_y = torch.tensor(y, dtype=torch.int64)
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        tra_dataloader = DataLoader(my_dataset, num_workers=1, batch_size=batch_size, shuffle=True, drop_last=False,
                                    pin_memory=True)
        return dataset_prop, tra_dataloader

    def fit(self, X, y, epochs=50, verbose=0, **kwd):
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
        y = np.random.choice(self.n_classes, X.shape[0])
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            a_label = a_label.to(self.device).squeeze()
            test_ = self.class_net(a_data, dataset_prop)
            _, predicted = torch.max(test_, 1)
        return predicted.detach().numpy()

    def score(self, X, y, sample_weight=None, verbose=0):
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

    # def score(self, X, y, sample_weight=None, verbose=0):
    #     return self.estimate_mi(X=X, y=y, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        y = np.random.choice(self.n_classes, X.shape[0])
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            test_ = self.class_net.score(a_data, dataset_prop)
        return test_.detach().numpy()

    def decision_function(self, X, verbose=0):
        y = np.random.choice(self.n_classes, X.shape[0])
        dataset_prop, test_dataloader = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0])
        for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
            a_data = a_data.to(self.device)
            test_ = self.class_net.score(a_data, dataset_prop)
        return test_.detach().numpy()

    def estimate_mi(self, X, y, verbose=1, **kwargs):
        dataset_prop, test_dataset = self.pytorch_tensor_dataset(X, y, batch_size=1)
        if verbose != 0:
            self.logger.info('MI estimation. ')
        softmax_list = []
        for a_data, a_label in test_dataset:
            int_label = a_label.cpu().item()
            a_data = a_data.unsqueeze(0).to(self.device)
            test_ = self.class_net(a_data, dataset_prop)
            test_1 = self.class_net.score(a_data, dataset_prop)
            if self.is_pc_softmax:
                a_softmax = torch.flatten(own_softmax(test_, dataset_prop, self.device))[int_label]
            else:
                a_softmax = torch.flatten(torch.softmax(test_, dim=-1))[int_label]
            if self.is_pc_softmax:
                softmax_list.append(math.log2(a_softmax.cpu().item()))
            else:
                softmax_list.append(math.log2(a_softmax.cpu().item()) + math.log2(len(dataset_prop)))
            if verbose != 0:
                self.logger.info("####################################################################################")
                self.logger.info(f"Score {test_1.detach().numpy()}, Test Score {test_.detach().numpy()}")
                self.logger.info(f"Data {a_data} Label {a_label}")
                self.logger.info(f"a_softmax {a_softmax} Label {a_label}")
                self.logger.info(f"Log Softmax {math.log(a_softmax.cpu().item())} Log M {math.log(len(dataset_prop))} "
                                 f"Label {int_label}")
                self.logger.info("####################################################################################")

        mi_estimated = np.nanmean(softmax_list)
        if verbose != 0:
            self.logger.error(f'Estimated MI: {mi_estimated}')
        if np.isnan(mi_estimated) or np.isinf(mi_estimated):
            if verbose != 0:
                self.logger.error(f'Setting MI to 0')
            mi_estimated = 0
        if self.mi_val - mi_estimated > .01:
            mi_estimated = self.mi_val
        mi_estimated = np.max([mi_estimated, 0.0])
        return mi_estimated
