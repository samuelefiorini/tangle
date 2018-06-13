"""Define keras layers."""
from keras import backend as K
from keras.engine.topology import Layer


# from: keras/examples/cifar10_cnn_capsule.py
# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
# Thanks this function transposing input is no longer needed
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


class ConvexCombination(Layer):
    def __init__(self, **kwargs):
        super(ConvexCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        _, n_hidden, _ = input_shape[0]
        # input_shape is:
        # [(None, recurrent_hidden_units, n_timespans),
        #  (None, recurrent_hidden_units, n_timespans)]
        # Adding one dimension for broadcasting
        self.lambd = self.add_weight(name='lambda',
                                     shape=(n_hidden, 1),
                                     initializer='glorot_uniform',
                                     trainable=True)
        super(ConvexCombination, self).build(input_shape)

    def call(self, x):
        # x is a list of two tensors with
        # shape=(batch_size, recurrent_hidden_units, n_timespans)
        return self.lambd * x[0] + (1 - self.lambd) * x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class NeuralAttention(Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        """Implementation of the standard attention layer.

        This implementation is inspired by
        ``https://github.com/philipperemy/keras-attention-mechanism``.

        Parameters:
        --------------
        units: int
            Dimensionality of the hidden attention space.

        use_bias: bool
            Whether the layer uses a bias vector.
        """
        self.units = units
        self.use_bias = use_bias
        self.output_dim = None
        super(NeuralAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is:
        # [(None, n_timespans, recurrent_hidden_units)]
        # Useful quantities
        self.n_timespans = input_shape[1]
        self.n_recurrent_hidden = input_shape[-1]

        # Dense MBS-items weights (tanh activation)
        self.kernel_x = self.add_weight(name='kernel_x',
                                        shape=(self.n_recurrent_hidden,
                                               self.units),
                                        initializer='glorot_uniform',
                                        trainable=True)

        # Dense weights (softmax activations)
        self.kernel_a = self.add_weight(name='kernel_a',
                                        shape=(self.units,
                                               self.n_recurrent_hidden),
                                        initializer='glorot_uniform',
                                        trainable=True)
        if self.use_bias:
                self.bias_x = self.add_weight(name='bias_x',
                                              shape=(self.units,),
                                              initializer='zeros',
                                              trainable=True)

        # The output dimension should be the same as the input one
        self.output_dim = input_shape
        super(NeuralAttention, self).build(input_shape)

    def call(self, input):
        x = input  # notation consistency

        # First two dense layers with linear activation
        gamma = K.dot(x, self.kernel_x)
        if self.use_bias:
            gamma = K.bias_add(gamma, self.bias_x)
        gamma = K.tanh(gamma)

        # Dense layer with softmax activation (no bias needed)
        alpha = softmax(K.dot(gamma, self.kernel_a), axis=-2)

        return alpha

    def compute_output_shape(self, input_shape):
        return self.output_dim


class TimespanGuidedNeuralAttention(Layer):
    def __init__(self, units=32, use_bias=True, **kwargs):
        """Implementation of the timespan-guided neural attention layer.

        Parameters:
        --------------
        units: int
            Dimensionality of the hidden attention space.

        use_bias: bool
            Whether the layer uses a bias vector.
        """
        self.units = units
        self.use_bias = use_bias
        self.output_dim = None
        super(TimespanGuidedNeuralAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is:
        # [(None, n_timespans, recurrent_hidden_units),
        #  (None, n_timespans, recurrent_hidden_units)]
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')
        if len(input_shape) < 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        timespans = [s[1] for s in input_shape if s is not None]
        if len(set(timespans)) > 1:
            raise ValueError('The two inputs should have the same number of '
                             'timespans. Got {}.'.format(timespans))

        # Other useful variables
        self.n_timespans = timespans[0]
        self.n_recurrent_hidden = input_shape[0][-1]

        # Dense MBS-items weights (tanh activation)
        self.kernel_x = self.add_weight(name='kernel_x',
                                        shape=(self.n_recurrent_hidden,
                                               self.units),
                                        initializer='glorot_uniform',
                                        trainable=True)
        # Dense timespan weights (tanh activation)
        self.kernel_t = self.add_weight(name='kernel_t',
                                        shape=(self.n_recurrent_hidden,
                                               self.units),
                                        initializer='glorot_uniform',
                                        trainable=True)
        # Dense weights (softmax activations)
        self.kernel_a = self.add_weight(name='kernel_a',
                                        shape=(self.units,
                                               self.n_recurrent_hidden),
                                        initializer='glorot_uniform',
                                        trainable=True)
        if self.use_bias:
                self.bias_x = self.add_weight(name='bias_x',
                                              shape=(self.units,),
                                              initializer='zeros',
                                              trainable=True)
                self.bias_t = self.add_weight(name='bias_t',
                                              shape=(self.units,),
                                              initializer='zeros',
                                              trainable=True)

        # The output dimension should be the same as the input one
        self.output_dim = input_shape[0]
        super(TimespanGuidedNeuralAttention, self).build(input_shape)

    def call(self, inputs):
        assert len(inputs) == 2
        x, t = inputs[0], inputs[1]  # define input

        # First two dense layers with linear activation
        gamma = K.dot(x, self.kernel_x)
        beta = K.dot(t, self.kernel_t)
        if self.use_bias:
            gamma = K.bias_add(gamma, self.bias_x)
            beta = K.bias_add(beta, self.bias_t)
        gamma = K.tanh(gamma)
        beta = K.tanh(beta)

        # Convex combination of the two resulting tensors
        # lambda * gamma + (1 - lambda) * beta
        delta = ConvexCombination()([gamma, beta])

        # Dense layer with softmax activation (no bias needed)
        alpha = softmax(K.dot(delta, self.kernel_a), axis=-2)

        return alpha

    def compute_output_shape(self, input_shape):
        return self.output_dim
