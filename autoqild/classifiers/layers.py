from keras.layers import Layer, Dense, Activation, BatchNormalization


class NormalizedDense(Layer):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use
        (see [activations](../activations.md)).
        If you don't specify anything, no activation is applied
        (ie. "relu:).
    normalize_before_activation: True if normalize the inputs before applying the activation.
    False if activation is applied before Bach Normalization
    """

    def __init__(self, units, activation="relu", **kwd):
        self.dense = Dense(units, activation="linear", **kwd)
        self.activation = Activation(activation=activation)
        self.batchnorm = BatchNormalization()
        self.norm_layer = None

    def __call__(self, x):
        return self.batchnorm(self.activation(self.dense(x)))

    def get_weights(self):
        w_b = self.batchnorm.get_weights()
        w_d = self.dense.get_weights()
        return w_b, w_d

    def set_weights(self, weights):
        w_b, w_d = weights
        self.batchnorm.set_weights(w_b)
        self.dense.set_weights(w_d)
