"""Configurations for search space for AutoGluon tools."""
from autogluon.common.space import Real, Int, Categorical

hyperparameters = {
    "NN_TORCH": {
        "learning_rate": Real(1e-5, 1e-1, default=5e-4, log=True),
        "dropout_prob": Real(0.0, 0.5, default=0.1),
        "num_layers": Int(lower=2, upper=20, default=5),
        "hidden_size": Int(lower=8, upper=256, default=32),
    },
    "GBM": {
        "n_estimators": Int(20, 300),
        "learning_rate": Real(1e-2, 0.5, log=True),
        "max_depth": Int(3, 20),
        "num_leaves": Int(20, 300),
        "feature_fraction": Real(0.2, 0.95, log=True),
        "bagging_fraction": Real(0.2, 0.95, log=True),
        "min_data_in_leaf": Int(20, 5000),
        "lambda_l1": Real(1e-6, 1e-2, log=True),
        "lambda_l2": Real(1e-6, 1e-2, log=True),
    },
    "CAT": {
        "learning_rate": Real(1e-2, 0.5, log=True),
        "depth": Int(4, 10),
        "l2_leaf_reg": Real(0.1, 10),
    },
    "XGB": {
        "n_estimators": Int(20, 300),
        "max_depth": Int(3, 10),
        "learning_rate": Real(1e-2, 0.5, log=True),
        "gamma": Real(0.0, 1.0),
        "reg_alpha": Real(0.0, 1.0),
        "reg_lambda": Real(0.0, 1.0),
    },
    "FASTAI": {
        "learning_rate": Real(1e-5, 1e-1, default=5e-4, log=True),
        "wd": Real(1e-6, 1e-1, default=5e-4, log=True),
        "emb_drop": Real(0.0, 0.5),
        "ps": Real(0.0, 0.5, ),
        "smoothing": Real(0.0, 0.5),
    },
    "RF": {
        "n_estimators": Int(20, 300),
        "criterion": Categorical("gini", "entropy"),
        "max_depth": Int(lower=6, upper=20, default=10),
        "max_features": Categorical("sqrt", "log2"),
        "min_samples_leaf": Int(lower=2, upper=50, default=10),
        "min_samples_split": Int(lower=2, upper=50, default=10),
        "class_weight": Categorical("balanced", "balanced_subsample")
    },
    "XT": {
        "n_estimators": Int(20, 300),
        "criterion": Categorical("gini", "entropy"),
        "max_depth": Int(lower=6, upper=20, default=10),
        "max_features": Categorical("sqrt", "log2"),
        "min_samples_leaf": Int(lower=2, upper=50, default=10),
        "min_samples_split": Int(lower=2, upper=50, default=10),
        "class_weight": Categorical("balanced", "balanced_subsample")
    },
    "KNN": {
        "weights": Categorical("uniform", "distance"),
        "n_neighbors": Int(lower=3, upper=5, default=5),
        "p": Categorical(1, 2, 3),
    },
}
"""This dictionary defines the hyperparameters for several models like
`NN_TORCH`, `GBM`, `CAT`, `XGB`, `FASTAI`, `RF`, `XT`, and `KNN`.

These models are commonly used in machine learning pipelines,
and each hyperparameter is configured using the `Real`, `Int`, or `Categorical` space from AutoGluon, which supports
hyperparameter tuning.
"""

reduced_hyperparameters = {
    "FASTAI": {
        "learning_rate": Real(1e-5, 1e-1, default=5e-4, log=True),
        "wd": Real(1e-6, 1e-1, default=5e-4, log=True),
        "emb_drop": Real(0.0, 0.5),
        "ps": Real(0.0, 0.5, ),
        "smoothing": Real(0.0, 0.5),
    },
    "RF": {
        "n_estimators": Int(20, 300),
        "criterion": Categorical("gini", "entropy"),
        "max_depth": Int(lower=6, upper=20, default=10),
        "max_features": Categorical("sqrt", "log2"),
        "min_samples_leaf": Int(lower=2, upper=50, default=10),
        "min_samples_split": Int(lower=2, upper=50, default=10),
        "class_weight": Categorical("balanced")
    },
    "XT": {
        "n_estimators": Int(20, 300),
        "criterion": Categorical("gini", "entropy"),
        "max_depth": Int(lower=6, upper=20, default=10),
        "max_features": Categorical("sqrt", "log2"),
        "min_samples_leaf": Int(lower=2, upper=50, default=10),
        "min_samples_split": Int(lower=2, upper=50, default=10),
        "class_weight": Categorical("balanced")
    },
}
"""This dictionary defines the hyperparameters for simpler models like
`FASTAI`, `RF` and `XT`.

These models are
commonly used in machine learning pipelines, and each hyperparameter is configured using the `Real`, `Int`, or
`Categorical` space from AutoGluon, which supports  hyperparameter tuning.
"""

# NN: `autogluon.tabular.models.tabular_nn.hyperparameters.parameters`
# Note: certain hyperparameter settings may cause these neural networks to train much slower.
# GBM: `autogluon.tabular.models.lgb.hyperparameters.parameters`
#  See also the lightGBM docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html
# CAT: `autogluon.tabular.models.catboost.hyperparameters.parameters`
#  See also the CatBoost docs: https://catboost.ai/docs/concepts/parameter-tuning.html
# XGB: `autogluon.tabular.models.xgboost.hyperparameters.parameters`
#  See also the XGBoost docs: https://xgboost.readthedocs.io/en/latest/parameter.html
# FASTAI: `autogluon.tabular.models.fastainn.hyperparameters.parameters`
#  See also the FastAI docs: https://docs.fast.ai/tabular.models.html
# RF: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Note: Hyperparameter tuning is disabled for this model.
# XT: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
# Note: Hyperparameter tuning is disabled for this model.
# KNN: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# Note: Hyperparameter tuning is disabled for this model.
# LR: `autogluon.tabular.models.lr.hyperparameters.parameters`
# Note: Hyperparameter tuning is disabled for this model.
# Note: `penalty` parameter can be used for regression to specify regularization method: `L1` and `L2` values are supported.
