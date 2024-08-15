"""A custom dense layer with batch normalization and optional pre-activation
normalization."""
from keras.layers import Layer, Dense, Activation, BatchNormalization


class NormalizedDense(Layer):
    """A custom dense layer with batch normalization and optional pre-
    activation normalization.

    Parameters
    ----------
    units : int
        Positive integer, dimensionality of the output space.
    activation : str, optional
        Activation function to use (default is `relu`).
        If not specified, no activation is applied (i.e., "linear").
    normalize_before_activation : bool, optional
        If True, normalizes the inputs before applying the activation function.
        If False, applies activation before batch normalization.
    **kwd : dict
        Additional keyword arguments to pass to the `Dense` layer.

    Attributes
    ----------
    dense : keras.layers.Dense
        Dense layer instance.
    activation : keras.layers.Activation
        Activation function to be applied.
    batchnorm : keras.layers.BatchNormalization
        Batch normalization layer.
    norm_layer : keras.layers.Layer or None
        Normalization layer if specified.
    """

    def __init__(self, units, activation="relu", **kwd):
        super(NormalizedDense, self).__init__()
        self.dense = Dense(units, activation="linear", **kwd)
        self.activation = Activation(activation=activation)
        self.batchnorm = BatchNormalization()
        self.norm_layer = None

    def __call__(self, x):
        batch_norm = self.batchnorm(self.activation(self.dense(x)))
        return batch_norm

    def get_weights(self):
        """Returns the weights of the batch normalization and dense layers.

        Returns
        -------
        w_b : tensor
            Weights of the batch normalization layers.
        w_d : tensor
            Weights of the dense layers.
        """
        w_b = self.batchnorm.get_weights()
        w_d = self.dense.get_weights()
        return w_b, w_d

    def set_weights(self, weights):
        """Sets the weights of the batch normalization and dense layers.

        Parameters
        ----------
        weights : tuple, (tensor, tensor)
            A tuple containing the weights for the batch normalization and dense layers.
        """
        w_b, w_d = weights
        self.batchnorm.set_weights(w_b)
        self.dense.set_weights(w_d)
