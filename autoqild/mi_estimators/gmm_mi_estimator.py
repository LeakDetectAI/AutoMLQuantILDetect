"""Gaussian Mixture Model-based MI estimator for evaluating mutual information
using probabilistic clustering."""

import copy
import logging

import numpy as np
from infoselect import get_gmm, SelectVars
from sklearn.linear_model import LogisticRegression

from autoqild.mi_estimators.mi_base_class import MIEstimatorBase
from ..utilities import create_dimensionality_reduction_model, log_exception_error


class GMMMIEstimator(MIEstimatorBase):
    """GMMMIEstimator class for estimating Mutual Information (MI) using
    Gaussian Mixture Models (GMMs) and performing classification using Logistic
    Regression.

    This class leverages GMMs to estimate mutual information and uses feature reduction techniques
    to create a robust classification model. It evaluates different GMMs based on goodness-of-fit measures
    such as AIC, BIC, and log-likelihood.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification data samples.

    n_features : int
        Number of features or dimensionality of the inputs of the classification data samples.

    y_cat : bool, optional, default=False
        Indicates if the target variable should be considered categorical or real-valued.

    covariance_type : {`full`, `tied`, `diag`, `spherical`}, default=`full`
        String describing the type of covariance parameters to use. Must be one of:

        - `full`: each component has its own general covariance matrix.
        - `tied`: all components share the same general covariance matrix.
        - `diag`: each component has its own diagonal covariance matrix.
        - `spherical`: each component has its own single variance.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance. Ensures that the covariance matrices are all positive.

    val_size : float, optional, default=0.30
        Validation set size as a proportion of the dataset to estimate GMMs.

    n_reduced : int, optional, default=20
        Number of features to reduce to in case n_features > 100.

    reduction_technique : str, optional, default=`select_from_model_rf`
        Technique to use for feature reduction, provided by scikit-learn.
        Must be one of:

        - `recursive_feature_elimination_et`: Uses ExtraTreesClassifier to recursively remove features and build a model.
        - `recursive_feature_elimination_rf`: Uses RandomForestClassifier to recursively remove features and build a model.
        - `select_from_model_et`: Meta-transformer for selecting features based on importance weights using ExtraTreesClassifier.
        - `select_from_model_rf`: Meta-transformer for selecting features based on importance weights using RandomForestClassifier.
        - `pca`: Principal Component Analysis for dimensionality reduction.
        - `lda`: Linear Discriminant Analysis for separating classes.
        - `tsne`: t-Distributed Stochastic Neighbor Embedding for visualization purposes.
        - `nmf`: Non-Negative Matrix Factorization for dimensionality reduction.

    random_state : int or object, optional, default=42
        Random state for reproducibility.

    **kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    ----------
    y_cat : bool
        Indicates if the target variable should be considered categorical or real-valued.

    num_comps : list
        List of component counts for GMM evaluation.

    reg_covar : float
        Regularization parameter for the GMM covariance matrices.

    n_models : int
        Number of GMM models to fit and evaluate.

    covariance_type : str
        The covariance type for the GMM.

    val_size : float
        Validation set size as a proportion of the dataset.

    n_reduced : int
        Number of reduced features for dimensionality reduction.

    reduction_technique : str
        Technique used for feature reduction.

    selection_model : object or None
        The fitted feature selection model, or None if not yet fitted.

    __is_fitted__ : bool
        Indicates whether the model is fitted.

    cls_model : LogisticRegression
        The classification model used after feature reduction.

    best_model : object or None
        The best fitted GMM model based on likelihood, or None if no model is selected.

    best_gmm_model : object or None
        The best fitted GMM used for mutual information estimation.

    best_likelihood : float or None
        The highest log-likelihood score achieved during model evaluation.

    best_bic : float or None
        The best Bayesian Information Criterion (BIC) score.

    best_aic : float or None
        The best Akaike Information Criterion (AIC) score.

    best_mi : float or None
        The best estimated mutual information.

    best_seed : int or None
        The random seed used to achieve the best model.

    round : int or None
        The optimal round for feature selection.

    logger : logging.Logger
        Logger instance for logging information.

    Private Methods
    ---------------
    __get_goodnessof_fit__(gmm, X, y):
        Calculate goodness of fit for the GMM model(s) used for MI estimation using Gaussian Mixture Models (GMMs).

    __transform__(X, y=None):
        Transform and reduce the feature matrix with 'n_features' features, using the specified reduction
        technique to the feature matrix with 'n_reduced' features.
    """

    def __init__(
        self,
        n_classes,
        n_features,
        y_cat=False,
        covariance_type="full",
        reg_covar=1e-06,
        val_size=0.30,
        n_reduced=20,
        reduction_technique="select_from_model_rf",
        random_state=42,
        **kwargs,
    ):
        super().__init__(n_classes=n_classes, n_features=n_features, random_state=random_state)
        self.y_cat = y_cat
        self.num_comps = list(np.arange(2, 20, 2))
        self.reg_covar = reg_covar
        self.n_models = 5
        self.covariance_type = covariance_type
        self.val_size = val_size
        if n_reduced > n_features:
            self.logger.warning(
                f"Reduced features {n_reduced} are less than actual features {n_features}"
            )
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

    def __get_goodnessof_fit__(self, gmm, X, y):
        """Calculate goodness of fit for the GMM model(s) used for estimating
        the Mutual Information (MI) using Gaussian Mixture Models (GMMs).

        Parameters
        ----------
        gmm : GMM or dict
            Gaussian Mixture Model or dictionary of GMMs.

        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        aic_fit : float
            Akaike information criterion for the current model on the input X.

        bic_fit : float
            Bayesian information criterion for the current model on the input X.

        likelihood : float
            Compute the per-sample average log-likelihood of the given data X.

        n_components : int
            Number of components in the GMM.
        """
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
        self.logger.info(f"AIC: {aic_fit}, BIC: {bic_fit}, Likelihood score {likelihood}")
        return aic_fit, bic_fit, likelihood, n_components

    def __transform__(self, X, y=None):
        """Transform and reduce the feature matrix with 'n_features' features,
        using the specified reduction technique to the feature matrix with
        'n_reduced' features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,), optional
            Target vector.

        Returns
        -------
        X : array-like of shape (n_samples, n_reduced)
            Transformed feature matrix.
        """
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
            self.selection_model = create_dimensionality_reduction_model(
                reduction_technique=self.reduction_technique, n_reduced=self.n_reduced
            )
            self.logger.info(f"Creating the model")
            if self.n_features > 50 and self.n_reduced < self.n_features:
                self.logger.info(
                    f"Transforming and reducing the {self.n_features} features to {self.n_reduced}"
                )
                self.selection_model.fit(X, y)
                X = self.selection_model.transform(X)
                self.__is_fitted__ = True
        else:
            if self.n_features > 50 and self.n_reduced < self.n_features:
                X = self.selection_model.transform(X)
        self.logger.info(f"After transform n_instances {X.shape[0]} n_features {X.shape[-1]}")
        return X

    def fit(self, X, y, verbose=0, **kwd):
        """Fit the GMM model and estimate mutual information.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        verbose : int, optional, default=0
            print or not to print!?.

        **kwd : dict, optional
            Additional keyword arguments.

        Returns
        -------
        self : GMMMIEstimator
            Fitted estimator.
        """
        X = self.__transform__(X, y)
        self.best_likelihood = -np.inf
        seed = self.random_state.randint(2**31, dtype="uint32")
        for iter_ in range(self.n_models):
            # self.logger.info(f"++++++++++++++++++ GMM Model {iter_} ++++++++++++++++++")
            try:
                gmm = get_gmm(
                    X,
                    y,
                    covariance_type=self.covariance_type,
                    y_cat=self.y_cat,
                    num_comps=self.num_comps,
                    reg_covar=self.reg_covar,
                    val_size=self.val_size,
                    random_state=seed + iter_,
                )
                self.logger.info(f"GMM Model {gmm}")
                select = SelectVars(gmm, selection_mode="backward")
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = (
                    select.get_info().values[0][1],
                    select.get_info().values[0][2],
                )
                mi = np.max([mi_mean, 0.0]) * np.log2(np.e)
                if not (np.isnan(mi) or np.isinf(mi)):
                    aic, bic, likelihood, n_components = self.__get_goodnessof_fit__(gmm, X, y)
                    # self.logger.info(f"MI {np.around(mi, 4)}  BIC {np.around(bic, 4)} Likelihood "
                    #                 f"{np.around(likelihood, 4)} n_components {n_components}")
                    if self.best_likelihood < likelihood:
                        self.logger.info(
                            f"GMM Model {iter_} set best with likelihood {np.around(likelihood, 4)} "
                            f"AIC {np.around(aic, 4)} BIC {np.around(bic, 4)} MI {np.around(mi, 4)}"
                        )
                        self.best_likelihood = likelihood
                        self.best_bic = bic
                        self.best_aic = aic
                        self.best_mi = mi
                        self.best_model = copy.deepcopy(select)
                        self.best_seed = seed + iter_
                        self.best_gmm_model = get_gmm(
                            X,
                            y,
                            covariance_type=self.covariance_type,
                            y_cat=self.y_cat,
                            num_comps=self.num_comps,
                            reg_covar=self.reg_covar,
                            val_size=self.val_size,
                            random_state=seed + iter_,
                        )
                else:
                    self.logger.info(f"Model {iter_} trained estimates wrong MI")
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Model {iter_} was not valid ")
            # self.logger.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        self.create_classification_model(X, y)
        return self

    def create_classification_model(self, X, y, **kwd):
        """Create the logistic regression classification model on reduced
        feature space with n_reduced features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        **kwd : dict, optional
            Additional keyword arguments.
        """
        self.logger.debug(f"Best Model is not None out of {self.n_models} seed {self.best_seed}")
        X = self.__transform__(X, y)
        if self.best_model is not None:
            idx = np.where(self.best_model.get_info()["delta"].values < 0)
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
        """Predict class labels for the input samples with reduced features of
        n_reduced using the fitted logistic regression classification model.

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
        X = self.__transform__(X)
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.predict(X=X)

    def score(self, X, y, sample_weight=None, verbose=0):
        """Compute the likelihood score of the GMM model.

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
            The score of the model based on likelihood.
        """
        X = self.__transform__(X, y)
        try:
            aic, bic, likelihood, n_components = self.__get_goodnessof_fit__(
                self.best_model.gmm, X, y
            )
            mi_mean, _ = (
                self.best_model.get_info().values[0][1],
                self.best_model.get_info().values[0][2],
            )
            mi = np.max([mi_mean, 0.0]) * np.log2(np.e)
            self.logger.info(
                f"MI {np.around(mi, 4)}  AIC {np.around(aic, 4)} BIC {np.around(bic, 4)} "
                f"Likelihood {np.around(likelihood, 4)} n_components {n_components}"
            )
            score = likelihood
            self.logger.debug(f"Best Model is not None out of {self.n_models} score {score}")
        except Exception as error:
            self.logger.debug("Best Model is None")
            log_exception_error(self.logger, error)
            score = -1000000
        return score

    def predict_proba(self, X, verbose=0):
        """Predict class labels for the input samples with reduced features of
        n_reduced using the fitted logistic regression classification model.

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
        X = self.__transform__(X)
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        y_pred = self.cls_model.predict_proba(X=X)
        return y_pred

    def decision_function(self, X, verbose=0):
        """Predict confidence scores for samples, which is proportional to the
        signed distance of that sample to the hyperplane.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        verbose : int, optional, default=0
            Verbosity level.

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values.
        """
        X = self.__transform__(X)
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.decision_function(X=X)

    def estimate_mi(self, X, y, verbose=0, **kwd):
        """Estimate mutual information using the best fitted GMM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        verbose : int, optional, default=0
            Verbosity level.

        **kwd : dict, optional
            Additional keyword arguments.

        Returns
        -------
        mi_estimated : float
            Estimated mutual information.
        """
        X = self.__transform__(X, y)
        iter_ = 0
        while True:
            try:
                iter_ += 1
                select = SelectVars(self.best_gmm_model, selection_mode="backward")
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = (
                    select.get_info().values[0][1],
                    select.get_info().values[0][2],
                )
                mi_estimated = np.nanmax([mi_mean, 0.0]) * np.log2(np.e)
                if verbose:
                    print(f"Model Number: {iter_}, Estimated MI: {mi_estimated}")
                self.logger.info(f"Model Number: {iter_}, Estimated MI: {mi_estimated}")
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Model {iter_} was not valid re-estimating it")
                mi_estimated = np.nan
            if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                self.logger.error(f"Nan MI Re-estimating")
            else:
                break
            if iter_ > 100:
                if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                    self.logger.error(f"Setting Mi to 0")
                    mi_estimated = 0.0
                break
        return mi_estimated
