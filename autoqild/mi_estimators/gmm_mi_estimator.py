import copy
import logging

import numpy as np
from infoselect import get_gmm, SelectVars
from sklearn.linear_model import LogisticRegression

from .mi_base_class import MIEstimatorBase
from ..utilities import create_dimensionality_reduction_model, log_exception_error


class GMMMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, n_features, y_cat=False, covariance_type='full', reg_covar=1e-06, val_size=0.30,
                 n_reduced=20, reduction_technique='select_from_model_rf', random_state=42, **kwargs):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state)
        self.y_cat = y_cat
        self.num_comps = list(np.arange(2, 20, 2))
        self.reg_covar = reg_covar
        self.n_models = 5
        self.covariance_type = covariance_type
        self.val_size = val_size

        self.n_reduced = n_reduced
        self.reduction_technique = reduction_technique
        self.selection_model = None
        self.__is_fitted__ = False

        # Classification Model
        self.cls_model = None
        self.best_model = None
        self.best_gmm_model = None
        self.best_likelihood = None
        self.best_bic = None
        self.best_aic = None
        self.best_mi = None
        self.best_seed = None
        self.round = None
        self.logger = logging.getLogger(GMMMIEstimator.__name__)

    def get_goodnessof_fit(self, gmm, X, y):
        if isinstance(gmm, dict):
            classes = list(set(y))
            bic_fit = []
            likelihood = []
            n_components = []
            aic_fit = []
            for c in classes:
                bic_fit.append(gmm[c].bic(X[y == c]))
                aic_fit.append(gmm[c].aic(X[y == c]))
                likelihood.append(gmm[c].score(X[y == c]))
                n_components.append(gmm[c].n_components)
            bic_fit = np.sum(bic_fit)
            aic_fit = np.sum(aic_fit)
            likelihood = np.mean(likelihood)
            n_components = np.mean(n_components)
        else:
            Z = np.hstack((y.reshape((-1, 1)), X))
            bic_fit = gmm.bic(Z)
            aic_fit = gmm.aic(Z)
            likelihood = gmm.score(Z)
            n_components = gmm.n_components
        return aic_fit, bic_fit, likelihood, n_components

    def transform(self, X, y=None):
        self.logger.info(f"Before transform n_instances {X.shape[0]} n_features {X.shape[-1]}")
        if y is not None:
            classes, n_classes = np.unique(y, return_counts=True)
            self.logger.info(f"Classes {classes} No of Classes {n_classes}")
        if not self.__is_fitted__:
            if self.n_features != X.shape[-1]:
                raise ValueError(f"Dataset passed does not contain {self.n_features}")
            if y is not None:
                if self.n_classes != len(np.unique(y)):
                    raise ValueError(f"Dataset passed does not contain {self.n_classes}")
            self.selection_model = create_dimensionality_reduction_model(reduction_technique=self.reduction_technique,
                                                                         n_reduced=self.n_reduced)
            self.logger.info(f"Creating the model")
            if self.n_features > 50 and self.n_reduced < self.n_features:
                self.logger.info(f"Transforming and reducing the {self.n_features} features to {self.n_reduced}")
                self.selection_model.fit(X, y)
                X = self.selection_model.transform(X)
                self.__is_fitted__ = True
        else:
            if self.n_features > 50 and self.n_reduced < self.n_features:
                X = self.selection_model.transform(X)
        self.logger.info(f"After transform n_instances {X.shape[0]} n_features {X.shape[-1]}")
        return X

    def fit(self, X, y, verbose=0, **kwd):
        X = self.transform(X, y)
        self.best_likelihood = -np.inf
        seed = self.random_state.randint(2 ** 31, dtype="uint32")
        for iter_ in range(self.n_models):
            # self.logger.info(f"++++++++++++++++++ GMM Model {iter_} ++++++++++++++++++")
            try:
                gmm = get_gmm(X, y, covariance_type=self.covariance_type, y_cat=self.y_cat, num_comps=self.num_comps,
                              reg_covar=self.reg_covar, val_size=self.val_size, random_state=seed + iter_)
                self.logger.info(f"GMM Model {gmm}")
                select = SelectVars(gmm, selection_mode='backward')
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = select.get_info().values[0][1], select.get_info().values[0][2]
                mi = np.max([mi_mean, 0.0]) * np.log2(np.e)
                if not (np.isnan(mi) or np.isinf(mi)):
                    aic, bic, likelihood, n_components = self.get_goodnessof_fit(gmm, X, y)
                    # self.logger.info(f"MI {np.around(mi, 4)}  BIC {np.around(bic, 4)} Likelihood "
                    #                 f"{np.around(likelihood, 4)} n_components {n_components}")
                    if self.best_likelihood < likelihood:
                        self.logger.info(f"GMM Model {iter_} set best with likelihood {np.around(likelihood, 4)} "
                                         f"AIC {np.around(aic, 4)} BIC {np.around(bic, 4)} MI {np.around(mi, 4)}")
                        self.best_likelihood = likelihood
                        self.best_bic = bic
                        self.best_aic = aic
                        self.best_mi = mi
                        self.best_model = copy.deepcopy(select)
                        self.best_seed = seed + iter_
                        self.best_gmm_model = get_gmm(X, y, covariance_type=self.covariance_type, y_cat=self.y_cat,
                                                      num_comps=self.num_comps, reg_covar=self.reg_covar,
                                                      val_size=self.val_size, random_state=seed + iter_)
                else:
                    self.logger.info(f"Model {iter_} trained estimates wrong MI")
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Model {iter_} was not valid ")
            # self.logger.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        self.create_classification_model(X, y)
        return self

    def create_classification_model(self, X, y, **kwd):
        self.logger.debug(f"Best Model is not None out of {self.n_models} seed {self.best_seed}")
        if self.best_model is not None:
            idx = np.where(self.best_model.get_info()['delta'].values < 0)
            try:
                # self.logger.info(self.best_model.get_info())
                # self.logger.info(f"Indices {idx[0]}")
                self.round = idx[0][0] - 1
            except IndexError as error:
                # log_exception_error(self.logger, error)
                self.round = 0
            X_new = self.best_model.transform(X, rd=self.round)
            self.cls_model = LogisticRegression()
            self.cls_model.fit(X_new, y)
        else:
            self.cls_model = LogisticRegression()
            self.cls_model.fit(X, y)

    def predict(self, X, verbose=0):
        X = self.transform(X)
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.predict(X=X)

    def score(self, X, y, sample_weight=None, verbose=0):
        X = self.transform(X, y)
        try:
            aic, bic, likelihood, n_components = self.get_goodnessof_fit(self.best_model.gmm, X, y)
            mi_mean, _ = self.best_model.get_info().values[0][1], self.best_model.get_info().values[0][2]
            mi = np.max([mi_mean, 0.0]) * np.log2(np.e)
            self.logger.info(f"MI {np.around(mi, 4)}  AIC {np.around(aic, 4)} BIC {np.around(bic, 4)} "
                             f"Likelihood {np.around(likelihood, 4)} n_components {n_components}")
            score = likelihood
            self.logger.debug(f"Best Model is not None out of {self.n_models} score {score}")
        except Exception as error:
            self.logger.debug("Best Model is None")
            log_exception_error(self.logger, error)
            score = -1000000
        return score

    def predict_proba(self, X, verbose=0):
        X = self.transform(X)
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.predict_proba(X=X)

    def decision_function(self, X, verbose=0):
        X = self.transform(X)
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.decision_function(X=X)

    def estimate_mi(self, X, y, verbose=0, **kwd):
        X = self.transform(X, y)
        iter_ = 0
        while True:
            try:
                iter_ += 1
                select = SelectVars(self.best_gmm_model, selection_mode='backward')
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = select.get_info().values[0][1], select.get_info().values[0][2]
                mi_estimated = np.nanmax([mi_mean, 0.0]) * np.log2(np.e)
                if verbose:
                    print(f'Model Number: {iter_}, Estimated MI: {mi_estimated}')
                self.logger.info(f'Model Number: {iter_}, Estimated MI: {mi_estimated}')
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Model {iter_} was not valid re-estimating it")
                mi_estimated = np.nan
            if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                self.logger.error(f'Nan MI Re-estimating')
            else:
                break
            if iter_ > 100:
                if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                    self.logger.error(f'Setting Mi to 0')
                    mi_estimated = 0.0
                break
        return mi_estimated
