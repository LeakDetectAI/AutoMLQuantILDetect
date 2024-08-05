import logging

import numpy as np
import torch
from pycilt.utils import softmax
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from .class_nn import StatNet
from .mi_base_class import MIEstimatorBase
from .pytorch_utils import get_optimizer_and_parameters, init, get_mine_loss


class MineMIEstimatorHPO(MIEstimatorBase):
    def __init__(self, n_classes, n_features, n_hidden=2, n_units=100, loss_function='donsker_varadhan_softplus',
                 optimizer_str='adam', learning_rate=1e-4, reg_strength=1e-10, encode_classes=True, random_state=42):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state)
        self.logger = logging.getLogger(MineMIEstimatorHPO.__name__)
        self.optimizer_str = optimizer_str
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.optimizer_cls, self._optimizer_config = get_optimizer_and_parameters(optimizer_str, learning_rate,
                                                                                  reg_strength)
        self.encode_classes = encode_classes
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.loss_function = loss_function
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.logger.info(
            f"device {self.device} cuda {torch.cuda.is_available()} gpu device {torch.cuda.device_count()}")
        self.optimizer = None
        self.stat_net = None
        self.dataset_properties = None
        self.label_binarizer = None
        self.final_loss = 0
        self.mi_val = 0

    def pytorch_tensor_dataset(self, X, y, batch_size=64, i=2):
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
        indices = rs.choice(xy_tilde.shape[0], size=batch_size)
        xy = xy[indices]
        xy_tilde = xy_tilde[indices]
        tensor_xy = torch.tensor(xy, dtype=torch.float32).to(self.device)  # transform to torch tensor
        tensor_xy_tilde = torch.tensor(xy_tilde, dtype=torch.float32).to(self.device)
        return tensor_xy, tensor_xy_tilde

    def fit(self, X, y, epochs=10000, batch_size=128, verbose=0, **kwd):
        MON_FREQ = epochs // 10
        # Monitoring
        MON_ITER = 10
        if self.encode_classes:
            y_t = LabelBinarizer().fit_transform(y)
            cls_enc = y_t.shape[-1]
        else:
            cls_enc = 1
        self.label_binarizer = LabelBinarizer().fit(y)
        self.stat_net = StatNet(in_dim=self.n_features, cls_enc=cls_enc, n_hidden=self.n_hidden, n_units=self.n_units,
                                device=self.device)
        self.stat_net.apply(init)
        self.stat_net.to(self.device)
        self.optimizer = self.optimizer_cls(self.stat_net.parameters(), **self._optimizer_config)
        all_estimates = []
        sum_loss = 0
        for iter_ in tqdm(range(epochs), total=epochs, desc='iteration'):
            self.stat_net.zero_grad()
            # print(f"iter {iter_}, y {y}")
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=batch_size, i=iter_)
            preds_xy = self.stat_net(xy)
            preds_xy_tilde = self.stat_net(xy_tilde)
            train_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
            loss = train_div.mul_(-1.)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss
            if (iter_ % MON_FREQ == 0) or (iter_ + 1 == epochs):
                with torch.no_grad():
                    mi_hats = []
                    for _ in range(MON_ITER):
                        # print(f"iter {iter_}, y {y}")
                        xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=batch_size, i=iter_)
                        preds_xy = self.stat_net(xy)
                        preds_xy_tilde = self.stat_net(xy_tilde)
                        eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                        mi_hats.append(eval_div.cpu().numpy())
                    mi_hat = np.mean(mi_hats)
                    if verbose:
                        print(f'iter: {iter_}, MI hat: {mi_hat} Loss: {loss.cpu().detach().numpy()[0]}')
                    self.logger.info(f'iter: {iter_}, MI hat: {mi_hat} Loss: {loss.cpu().detach().numpy()[0]}')
                    all_estimates.append(mi_hat)
        self.final_loss = sum_loss.cpu().detach().numpy()[0]
        mis = np.array(all_estimates)
        n = int(len(all_estimates) / 3)
        self.mi_val = np.nanmean(mis[np.argpartition(mis, -n)[-n:]])
        torch.no_grad()
        self.logger.info(f"Fit Loss {self.final_loss} MI Val: {self.mi_val}")
        return self

    def predict(self, X, verbose=0):
        scores = self.predict_proba(X=X, verbose=verbose)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        torch.no_grad()
        xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0], i=0)
        preds_xy = self.stat_net(xy).cpu().detach().numpy().flatten()
        preds_xy_tilde = self.stat_net(xy_tilde).cpu().detach().numpy().flatten()
        mse = mean_squared_error(preds_xy, preds_xy_tilde)
        self.logger.info(f"MSE {mse}")
        self.logger.info(f"Memory allocated {torch.cuda.memory_allocated()} Cached {torch.cuda.memory_cached()}")
        return mse

    def predict_proba(self, X, verbose=0):
        scores = self.decision_function(X=X, verbose=verbose)
        scores = softmax(scores)
        return scores

    def decision_function(self, X, verbose=0):
        scores = None
        for n_class in range(self.n_classes):
            y = np.zeros(X.shape[0]) + n_class
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0], i=0)
            score = self.stat_net(xy).cpu().detach().numpy()
            # self.logger.info(f"Class {n_class} scores {score.flatten()}")
            if scores is None:
                scores = score
            else:
                scores = np.hstack((scores, score))
        return scores

    def estimate_mi(self, X, y, verbose=0, MON_ITER=100):
        mi_hats = []
        for iter_ in range(MON_ITER):
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, batch_size=X.shape[0], i=iter_)
            preds_xy = self.stat_net(xy)
            preds_xy_tilde = self.stat_net(xy_tilde)
            eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
            mi_hat = eval_div.cpu().detach().numpy().flatten()[0]
            if verbose:
                print(f'iter: {iter_}, MI hat: {mi_hat}')
            mi_hats.append(mi_hat)
        mi_hats = np.array(mi_hats)
        n = int(MON_ITER / 2)
        mi_hats = mi_hats[np.argpartition(mi_hats, -n)[-n:]]
        mi_estimated = np.nanmean(mi_hats)
        if np.isnan(mi_estimated) or np.isinf(mi_estimated):
            self.logger.error(f'Setting MI to 0')
            mi_estimated = 0
        self.logger.info(f'Estimated MIs: {mi_hats[-10:]} Mean {mi_estimated}')
        if self.mi_val - mi_estimated > .01:
            mi_estimated = self.mi_val
        mi_estimated = np.max([mi_estimated, 0.0])
        return mi_estimated
