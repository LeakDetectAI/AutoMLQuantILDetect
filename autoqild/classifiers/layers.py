from keras.layers import Layer, Dense, Activation, BatchNormalization

class NormalizedDense(Layer):

    def __init__(self, units, activation="relu", **kwd):
        """
            A custom dense layer with batch normalization and optional pre-activation normalization.

            Parameters
            ----------
            units : int
                Positive integer, dimensionality of the output space.
            activation : str, optional
                Activation function to use (default is 'relu').
                If not specified, no activation is applied (i.e., "linear").
            normalize_before_activation : bool, optional
                If True, normalizes the inputs before applying the activation function.
                If False, applies activation before batch normalization.
            **kwd : dict
                Additional keyword arguments to pass to the `Dense` layer.
        """
        self.dense = Dense(units, activation="linear", **kwd)
        self.activation = Activation(activation=activation)
        self.batchnorm = BatchNormalization()
        self.norm_layer = None

    def __call__(self, x):
        """
            Applies the layer to the input tensor `x`.

            Parameters
            ----------
            x : tensor
                Input tensor to the layer.

            Returns
            -------
            tensor
                Output tensor after applying dense, activation, and batch normalization.
        """
        return self.batchnorm(self.activation(self.dense(x)))

    def get_weights(self):
        """
            Returns the weights of the batch normalization and dense layers.

            Returns
            -------
            tuple
                A tuple containing the weights of the batch normalization and dense layers.
        """
        w_b = self.batchnorm.get_weights()
        w_d = self.dense.get_weights()
        return w_b, w_d

    def set_weights(self, weights):
        """
            Sets the weights of the batch normalization and dense layers.

            Parameters
            ----------
            weights : tuple
                A tuple containing the weights for the batch normalization and dense layers.
        """
        w_b, w_d = weights
        self.batchnorm.set_weights(w_b)
        self.dense.set_weights(w_d)
