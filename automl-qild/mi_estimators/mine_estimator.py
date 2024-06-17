import logging
from itertools import product

import numpy as np
import torch
from pycilt.utils import softmax
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from .class_nn import StatNet
from .mi_base_class import MIEstimatorBase
from .pytorch_utils import get_optimizer_and_parameters, init, get_mine_loss


class MineMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, n_features, loss_function='donsker_varadhan_softplus', optimizer_str='adam',
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.logger.info(f"device {self.device} cuda {torch.cuda.is_available()} device {torch.cuda.device_count()}")
        self.optimizer = None
        self.dataset_properties = None
        self.label_binarizer = None
        self.final_loss = 0
        self.mi_validation_final = 0
        self.models = []
        self.n_models = 0

    def pytorch_tensor_dataset(self, X, y, i=2):
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
        MON_FREQ = epochs // 10
        # Monitoring
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
            for iter_ in tqdm(range(epochs), total=epochs, desc='iteration'):
                stat_net.zero_grad()
                # print(f"iter {iter_}, y {y}")
                xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=iter_)
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
                            # print(f"iter {iter_}, y {y}")
                            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=iter_)
                            preds_xy = stat_net(xy)
                            preds_xy_tilde = stat_net(xy_tilde)
                            eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                            mi_hats.append(eval_div.cpu().numpy())
                        mi_hat = np.mean(mi_hats)
                        if verbose:
                            print(f'iter: {iter_}, MI hat: {mi_hat} Loss: {loss.detach().numpy()[0]}')
                        self.logger.info(f'iter: {iter_}, MI hat: {mi_hat} Loss: {loss.detach().numpy()[0]}')
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
        scores = self.predict_proba(X=X, verbose=verbose)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        # mi = self.estimate_mi(X=X, y=y, verbose=verbose, MON_ITER=10)
        mi = self.mi_validation_final
        self.logger.info(f"Loss {self.final_loss} MI Val: {self.mi_validation_final}")
        if np.isnan(mi) or np.isinf(mi):
            mi = 0.0
        return mi

    def predict_proba(self, X, verbose=0):
        scores = self.decision_function(X=X, verbose=verbose)
        scores = softmax(scores)
        return scores

    def decision_function(self, X, verbose=0):
        scores = None
        final_scores = None
        for model in self.models:
            for n_class in range(self.n_classes):
                y = np.zeros(X.shape[0]) + n_class
                xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=0)
                score = model(xy).detach().numpy()
                # self.logger.info(f"Class {n_class} scores {score.flatten()}")
                if scores is None:
                    scores = score
                else:
                    scores = np.hstack((scores, score))
            final_scores += scores
        final_scores = final_scores / self.n_models
        return final_scores

    def estimate_mi(self, X, y, verbose=0, MON_ITER=1000):
        final_mis = []
        for model in self.models:
            mi_hats = []
            for iter_ in range(MON_ITER):
                xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=iter_)
                preds_xy = model(xy)
                preds_xy_tilde = model(xy_tilde)
                eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                mi_hat = eval_div.detach().numpy().flatten()[0]
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
            mi_estimated = np.max([mi_estimated, 0.0])
            final_mis.append(mi_estimated)
        final_mi = np.nanmedian(final_mis)
        final_mi = np.nanmax([final_mi, 0.0])
        return final_mi
