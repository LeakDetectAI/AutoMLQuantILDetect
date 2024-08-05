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
        if y is not None:
            y = to_categorical(y, num_classes=self.n_classes)
        return y

    def fit(self, X, y, epochs=50, batch_size=32, callbacks=None, validation_split=0.1, verbose=1, **kwd):
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
        scores = self.model.predict(x=X, verbose=verbose)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.predict(X, verbose=verbose)
        acc = balanced_accuracy_score(y, y_pred)
        return acc

    def predict_proba(self, X, verbose=0):
        prob_predictions = self.model.predict(x=X, verbose=verbose)
        return prob_predictions

    def decision_function(self, X, verbose=0):
        prob_predictions = self.scoring_model.predict(x=X, verbose=verbose)
        if self.n_classes == 2:
            prob_predictions = prob_predictions[:, 1]
        return prob_predictions

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
