import logging

import numpy as np
from keras import Input, Model
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils import class_weight

from .layers import NormalizedDense


class MultiLayerPerceptron(BaseEstimator, ClassifierMixin):
    """
        MultiLayerPerceptron class for building and training a feedforward neural network using Keras.

        Parameters
        ----------
        n_features : int
            Number of features or dimensionality of the inputs.

        n_classes : int
            Number of classes in the classification data samples.

        n_hidden : int, optional, default=10
            Number of hidden layers.

        n_units : int, optional, default=100
            Number of units per hidden layer.

        batch_normalization : bool, optional, default=True
            Whether to use batch normalization.

        activation : str, optional, default='relu'
            Activation function to use in the hidden layers.

        loss_function : str, optional, default='categorical_crossentropy'
            Loss function to use for training.

        metrics : list of str, optional, default=['accuracy']
            List of metrics to be evaluated by the model during training and testing.

        optimizer_str : {'adam', 'sgd', ...}, default='adam'
            Optimizer to use for training. Must be one of the optimizers available in Keras.

        reg_strength : float, optional, default=1e-4
            Regularization strength for the L2 regularizer.

        kernel_initializer : str, optional, default="lecun_normal"
            Initializer for the kernel weights matrix.

        learning_rate : float, optional, default=0.001
            Learning rate for the optimizer.

        early_stopping : bool, optional, default=False
            Whether to use early stopping during training.

        model_save_path : str, optional, default=''
            Path to save the trained model.

        random_state : int or None, optional, default=None
            Random state for reproducibility.

        **kwargs : dict, optional
            Additional keyword arguments.
    """

    def __init__(self, n_features, n_classes, n_hidden=10, n_units=100, batch_normalization=True, activation='relu',
                 loss_function='categorical_crossentropy', metrics=['accuracy'], optimizer_str='adam',
                 reg_strength=1e-4, kernel_initializer="lecun_normal", learning_rate=0.001,
                 early_stopping=False, model_save_path='', random_state=None, **kwargs):
        self.logger = logging.getLogger(name=MultiLayerPerceptron.__name__)
        self.n_features = n_features
        self.n_classes = n_classes
        self.classes_ = np.arange(0, self.n_classes)
        self.n_units = n_units
        self.n_hidden = n_hidden
        self.batch_normalization = batch_normalization
        if not self.batch_normalization:
            self.activation = 'selu'
        else:
            self.activation = activation
        self.loss_function = loss_function
        self.optimizer_str = optimizer_str
        if optimizer_str == 'adam':
            self.optimizer = optimizers.Adam()
        elif optimizer_str == 'sgd':
            self.optimizer = optimizers.SGD()
        else:
            self.optimizer = optimizers.get(optimizer_str)
        self._optimizer_config = self.optimizer.get_config()
        K.set_value(self.optimizer.lr, learning_rate)
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.model_save_path = model_save_path
        self.reg_strength = reg_strength
        self.kernel_regularizer = l2(l=self.reg_strength)
        self.kernel_initializer = kernel_initializer
        self.kwargs = kwargs
        self.random_state = check_random_state(random_state)
        self.model, self.scoring_model = None, None

    def _construct_layers(self, **kwargs):
        """
            Construct the input, hidden and output layers for the MLP model.
        """
        self.output_node = Dense(
            1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer
        )
        self.input = Input(shape=self.n_features, dtype='float32')
        if self.batch_normalization:
            self.hidden_layers = [
                NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in range(self.n_hidden)
            ]
        else:
            self.hidden_layers = [
                Dense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in range(self.n_hidden)
            ]
        self.score_layer = Dense(self.n_classes, activation=None, kernel_regularizer=self.kernel_regularizer)
        self.output_node = Activation('softmax', name='predictions')
        assert len(self.hidden_layers) == self.n_hidden

    def _construct_model_(self):
        """
            Construct and compile the Keras models.

            Returns
            -------
            model : keras.Model
                The compiled Keras model for training and prediction in form of labels.

            scoring_model : keras.Model
                The compiled Keras model for predicting real-valued scores.
        """
        x = self.hidden_layers[0](self.input)
        for hidden in self.hidden_layers[1:]:
            x = hidden(x)
            # x = BatchNormalization()(x)
        scores = self.score_layer(x)
        output = self.output_node(scores)
        model = Model(inputs=self.input, outputs=output, name="mlp_baseline")
        scoring_model = Model(inputs=self.input, outputs=scores, name="mlp_baseline_scorer")
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        scoring_model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        return model, scoring_model

    def reshape_inputs(self, y):
        """
            Reshape target variable to categorical format if necessary.

            Parameters
            ----------
            y : array-like of shape (n_samples,)
                Target vector.

            Returns
            -------
            y : array-like of shape (n_samples, n_classes)
                Reshaped target vector.
        """
        if y is not None:
            y = to_categorical(y, num_classes=self.n_classes)
        return y

    def fit(self, X, y, epochs=50, batch_size=32, callbacks=None, validation_split=0.1, verbose=1, **kwd):
        """
            Fit the MLP model to the training data.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Feature matrix.

            y : array-like of shape (n_samples,)
                Target vector.

            epochs : int, optional, default=50
                Number of training epochs.

            batch_size : 

            callbacks : list of keras.callbacks.Callback, optional
                List of callback instances to apply during training.

            validation_split : float, optional, default=0.1
                Fraction of the training data to be used as validation data.

            verbose : int, optional, default=1
                Verbosity mode.

            **kwd : dict, optional
                Additional keyword arguments.

            Returns
            -------
            self : MultiLayerPerceptron
                Fitted estimator.
        """
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = dict(enumerate(class_weights))
        self._construct_layers(
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            activation=self.activation,
            **self.kwargs
        )
        self.model, self.scoring_model = self._construct_model_()
        y = self.reshape_inputs(y)
        er = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        if self.early_stopping:
            if callbacks is not None:
                callbacks.append(er)
            else:
                callbacks = [er]
        self.model.fit(x=X, y=y, batch_size=batch_size, class_weight=class_weights, validation_split=validation_split,
                       epochs=epochs, callbacks=callbacks, verbose=verbose)

        return self

    def predict(self, X, verbose=0):
        """
            Predict class labels for the input samples with maximum class probability.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Feature matrix.

            verbose : int, optional, default=0
                Verbosity mode.

            Returns
            -------
            y_pred : array-like of shape (n_samples,)
                Predicted class labels.
        """
        scores = self.model.predict(x=X, verbose=verbose)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        """
            Compute the balanced accuracy score for the input samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Feature matrix.

            y : array-like of shape (n_samples,)
                True labels.

            sample_weight : array-like of shape (n_samples,), optional
                Sample weights.

            verbose : int, optional, default=0
                Verbosity mode.

            Returns
            -------
            acc : float
                Balanced accuracy score.
        """
        y_pred = self.predict(X, verbose=verbose)
        acc = balanced_accuracy_score(y, y_pred)
        return acc

    def predict_proba(self, X, verbose=0):
        """
            Predict class probabilities for the input samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Feature matrix.

            verbose : int, optional, default=0
                Verbosity mode.

            Returns
            -------
            p_pred : array-like of shape (n_samples, n_classes)
                Predicted class probabilities.
        """
        p_pred = self.model.predict(x=X, verbose=verbose)
        return p_pred

    def decision_function(self, X, verbose=0):
        """
           Compute the real valued scores for each class for the input samples.

           Parameters
           ----------
           X : array-like of shape (n_samples, n_features)
               Feature matrix.

           verbose : int, optional, default=0
               Verbosity mode.

           Returns
           -------
           decision : array-like of shape (n_samples,)
               Decision function values.
        """
        y_pred = self.scoring_model.predict(x=X, verbose=verbose)
        if self.n_classes == 2:
            y_pred = y_pred[:, 1]
        return y_pred

    def get_params(self, deep=True):
        """
            Get parameters for this estimator.

            Parameters
            ----------
            deep : bool, default=True
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

            Returns
            -------
            params : dict
                Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **parameters):
        """
            Set the parameters of this estimator.

            The method works on simple estimators as well as on nested objects
            (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
            parameters of the form ``<component>__<parameter>`` so that it's
            possible to update each component of a nested object.

            Parameters
            ----------
            **params : dict
                Estimator parameters.

            Returns
            -------
            self : estimator instance
                Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
