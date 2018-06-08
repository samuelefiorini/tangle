"""Bidirectional timestamp-guided attention model.

Keras implementation.
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import (LSTM, Bidirectional, Dense, Dot, Dropout,
                          Embedding, GlobalAveragePooling1D, Input, Multiply,
                          Permute)
from keras.models import Model
from keras.regularizers import l2

__implemeted_models__ = ['baseline', 'attention', 'proposed']


class ConvexCombination(Layer):
    def __init__(self, **kwargs):
        super(ConvexCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        _, n_hidden, _ = input_shape[0]
        # input_shape is:
        # [(None, recurrent_hidden_units, n_timestamps),
        #  (None, recurrent_hidden_units, n_timestamps)]
        # Adding one dimension for broadcasting
        self.lambd = self.add_weight(name='lambda',
                                     shape=(n_hidden, 1),
                                     initializer='glorot_uniform',
                                     trainable=True)
        super(ConvexCombination, self).build(input_shape)

    def call(self, x):
        # x is a list of two tensors with
        # shape=(batch_size, recurrent_hidden_units, n_timestamps)
        return self.lambd * x[0] + (1 - self.lambd) * x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(Layer):
    def __init__(self, use_bias=True, **kwargs):
        """Implementation of the standard attention layer.

        This implementation follows "Hierarchical Attention Networks for
        Document Classification" by Yang et al as closely as possible and it
        is inspired by
        https://github.com/philipperemy/keras-attention-mechanism.

        Parameters:
        --------------
        n_timestamps: int
            The number of input timestamps defines the size of the dense
            layers.
        """
        self.use_bias = use_bias
        self.output_dim = None
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is:
        # [(None, n_timestamps, recurrent_hidden_units)]
        # Useful quantities
        self.n_timestamps = input_shape[1]
        self.n_hidden = input_shape[-1]

        # Dense MBS-items weights (tanh activation)
        self.kernel_x = self.add_weight(name='kernel_x',
                                        shape=(self.n_timestamps,
                                               self.n_timestamps),
                                        initializer='glorot_uniform',
                                        trainable=True)
        # Dense weights (softmax activations)
        self.kernel_a = self.add_weight(name='kernel_a',
                                        shape=(self.n_timestamps,
                                               self.n_timestamps),
                                        initializer='glorot_uniform',
                                        trainable=True)
        if self.use_bias:
                self.bias_x = self.add_weight(name='bias_x',
                                              shape=(self.n_hidden,
                                                     self.n_timestamps),
                                              initializer='zeros',
                                              trainable=True)

        # The output dimension should be the same as the input one
        self.output_dim = input_shape
        super(Attention, self).build(input_shape)

    def call(self, input):
        x = Permute((2, 1))(input)  # transpose input

        # First two dense layers with linear activation
        gamma = K.dot(x, self.kernel_x)
        if self.use_bias:
            gamma = K.bias_add(gamma, self.bias_x)
        gamma = K.tanh(gamma)

        # Dense layer with softmax activation (no bias needed)
        alpha = K.softmax(K.dot(gamma, self.kernel_a))
        alpha = Permute((2, 1))(alpha)  # transpose back to the original shape

        return alpha

    def compute_output_shape(self, input_shape):
        return self.output_dim


class TimestampGuidedAttention(Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        """Implementation of the Timestamp guided attention layer.

        Parameters:
        --------------
        units: int
            Dimensionality of the hidden intermediate space.

        use_bias: bool (default = True)
            Whether the first dense layer uses a bias vector.
        """
        self.units = units
        self.use_bias = use_bias
        self.output_dim = None
        super(TimestampGuidedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is:
        # [(None, n_timestamps, recurrent_hidden_units),
        #  (None, n_timestamps, recurrent_hidden_units)]
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')
        if len(input_shape) < 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        timestamps = [s[1] for s in input_shape if s is not None]
        if len(set(timestamps)) > 1:
            raise ValueError('The two inputs should have the same number of '
                             'timestamps. Got {}.'.format(timestamps))

        # Other useful variables
        self.n_timestamps = timestamps[0]
        self.recurrent_hidden_units = input_shape[0][-1]

        # Dense MBS-items weights (tanh activation)
        # self.kernel_x = self.add_weight(name='kernel_x',
        #                                 shape=(self.units,
        #                                        self.recurrent_hidden_units),
        #                                 initializer='glorot_uniform',
        #                                 trainable=True)
        # Dense timestamp weights (tanh activation)
        # self.kernel_t = self.add_weight(name='kernel_t',
        #                                 shape=(self.units,
        #                                        self.recurrent_hidden_units),
        #                                 initializer='glorot_uniform',
        #                                 trainable=True)
        # Dense weights (softmax activations)
        # self.kernel_a = self.add_weight(name='kernel_a',
        #                                 shape=(self.n_timestamps,
        #                                        self.n_timestamps),
        #                                 initializer='glorot_uniform',
        #                                 trainable=True)
        # if self.use_bias:
        #         self.bias_x = self.add_weight(name='bias_x',
        #                                       shape=(self.units,
        #                                              self.n_timestamps),
        #                                       initializer='zeros',
        #                                       trainable=True)
        #         self.bias_t = self.add_weight(name='bias_t',
        #                                       shape=(self.units,
        #                                              self.n_timestamps),
        #                                       initializer='zeros',
        #                                       trainable=True)

        # The output dimension should be the same as the input one
        self.output_dim = input_shape[0]
        super(TimestampGuidedAttention, self).build(input_shape)

    def call(self, inputs):
        assert len(inputs) == 2
        # x = Permute((2, 1))(inputs[0])  # transpose input
        # t = Permute((2, 1))(inputs[1])
        x = inputs[0]  # transpose input
        t = inputs[1]

        print('hx: ', x.shape)
        print('tx: ', t.shape)

        # First two dense layers with linear activation
        gamma = Dense(self.units, use_bias=self.use_bias, activation='tanh')(x)
        beta = Dense(self.units, use_bias=self.use_bias, activation='tanh')(t)

        # gamma = K.dot(self.kernel_x, x)
        # print('gamma 1: ', gamma.shape)
        # beta = K.dot(t, self.kernel_t)
        # if self.use_bias:
        #     gamma = K.bias_add(gamma, self.bias_x)
        #     beta = K.bias_add(beta, self.bias_t)
        # gamma = K.tanh(gamma)
        # beta = K.tanh(beta)
        print('gamma: ', gamma.shape)
        print('beta: ', beta.shape)

        # Convex combination of the two resulting tensors
        # lambda * gamma + (1 - lambda) * beta
        delta = ConvexCombination()([gamma, beta])
        print('delta: ', delta.shape)

        # Dense layer with softmax activation (no bias needed)
        delta = Permute((2, 1))(delta)
        print('delta T: ', delta.shape)
        # alpha = Dense(self.n_timestamps, use_bias=False, activation='softmax')(delta)
        alpha = K.softmax(K.dot(delta, self.kernel_a))
        # alpha = K.softmax(delta)
        print('alpha: ', alpha.shape)
        alpha = Permute((2, 1))(alpha)  # transpose back to the original shape
        print('alpha T: ', alpha.shape)

        return alpha

    def compute_output_shape(self, input_shape):
        return self.output_dim


def build_model(mbs_input_shape, timestamp_input_shape, vocabulary_size,
                embedding_size=50, recurrent_units=8, attention_units=8,
                dense_units=8, bidirectional=True, LSTMLayer=LSTM):
    """Build keras proposed model.

    When the `bidirectional` flag is True, this function returns the
    bidirectional timestamp-guided attention model. Otherwise LSTMLayers flow
    forward only.

    Parameters:
    --------------
    mbs_input_shape: list
        Shape of the MBS sequence input. This parameter should be equal to
        `(maxlen,)` where `maxlen` is the maximum number of elements in a
        sequence.

    timestamp_input_shape: list
        Shape of the timestamp sequence input. It should be `(maxlen, 1)`.

    vocabulary_size: int
        The size of the tokenizer vocabulary. This parameter is used to build
        the embedding matrix.

    embedding_size: int (default=50)
        The size of the used word embedding, it should be in [50, 100, 200,
        300] according to `glove.6B`.

    recurrent_units: int (default=8)
        The number of recurrent units in the LSTM layers.

    attention_units: int (default=8)
        The number of intermediate hidden units of the
        TimestampGuidedAttention layer.

    dense_units: int (default=8)
        The number of dense units in the dense layers.

    bidirectional: bool (default=True)
        This flag control wether or bidirectional LSTM layers should be used.

    LSTMLayer: keras.layer(default=keras.layer.LSTM)
        This parameter controls which implementation of the LSTM layer should
         be used. There are two possibilities:
            - `keras.layer.LSTM`: slower implementation, that enables the use
              of recurrent dropout and the extraction of the activations by
              `mbspbs10pc.read_activations.get_activations()`
            - `keras.layer.CuDNNLSTM`: faster CuDNN implementation of the LSTM
              layer where recurrent dropout is not available nor the
              extraction of the activations.
        Best practice is to use the fast implementation for training and the
        slow one for attention extraction.
        For more info see https://keras.io/layers/recurrent/.

    Returns:
    --------------
    model: keras.Model object
        Keras implementation of the model.
    """
    # Channel 1: MBS
    mbs_input = Input(shape=mbs_input_shape, name='mbs_input')
    e = Embedding(vocabulary_size, embedding_size,
                  name='mbs_embedding')(mbs_input)
    if bidirectional:
        x1 = Bidirectional(LSTMLayer(recurrent_units, return_sequences=True),
                           name='mbs_lstm')(e)
    else:
        x1 = LSTMLayer(recurrent_units, return_sequences=True,
                       name='mbs_lstm')(e)

    # Channel 2: Timestamps
    timestamp_input = Input(shape=timestamp_input_shape,
                            name='timestamp_input')
    if bidirectional:
        x2 = Bidirectional(LSTMLayer(recurrent_units, return_sequences=True),
                           name='timestamp_lstm')(timestamp_input)
    else:
        x2 = LSTMLayer(recurrent_units, return_sequences=True,
                       name='timestamp_lstm')(timestamp_input)

    # -- Timestamp-guided attention -- #
    alpha = TimestampGuidedAttention(units=attention_units,
                                     name='tsg_attention')([x1, x2])
    # -- Timestamp-guided attention -- #

    # Combine channels to get contribution and context
    c = Multiply(name='contribution')([alpha, x1])
    x = Dot(axes=1, name='context')([c, e])

    # Output
    x = GlobalAveragePooling1D(name='pooling')(x)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='linear', name='fc')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='fc_output',
                   activity_regularizer=l2(0.002))(x)

    # Define the model
    model = Model(inputs=[mbs_input, timestamp_input],
                  outputs=[output])

    return model


def build_attention_model(mbs_input_shape, timestamp_input_shape,
                          vocabulary_size, embedding_size=50,
                          recurrent_units=8, dense_units=16,
                          bidirectional=True, LSTMLayer=LSTM):
    """Build keras attention model.

    This function builds a standard attention model, as in
    "Hierarchical Attention Networks for Document Classification"
    by Yang et al.

    Parameters:
    --------------
    see `build_model()`
    """
    # Channel 1: MBS
    mbs_input = Input(shape=mbs_input_shape, name='mbs_input')
    e = Embedding(vocabulary_size, embedding_size,
                  name='mbs_embedding')(mbs_input)
    if bidirectional:
        x1 = Bidirectional(LSTMLayer(recurrent_units, return_sequences=True),
                           name='mbs_lstm')(e)
    else:
        x1 = LSTMLayer(recurrent_units, return_sequences=True,
                       name='mbs_lstm')(e)

    # -- Timestamp-guided attention -- #
    alpha = Attention(name='tsg_attention')(x1)
    # -- Timestamp-guided attention -- #

    # Combine channels to get contribution and context
    c = Multiply(name='contribution')([alpha, x1])
    x = Dot(axes=1, name='context')([c, e])

    # Output
    x = GlobalAveragePooling1D(name='pooling')(x)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='linear', name='fc')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='fc_output',
                   activity_regularizer=l2(0.002))(x)

    # Define the model
    model = Model(inputs=[mbs_input],
                  outputs=[output])

    return model


def build_baseline_model(mbs_input_shape, timestamp_input_shape,
                         vocabulary_size, embedding_size=50, recurrent_units=8,
                         dense_units=16, bidirectional=True, LSTMLayer=LSTM):
    """Build keras baseline model.

    This function builds a baseline model made of Embedding + LSTM layers only.

    Parameters:
    --------------
    see `build_model()`
    """
    # Channel 1: MBS
    mbs_input = Input(shape=mbs_input_shape, name='mbs_input')
    e = Embedding(vocabulary_size, embedding_size,
                  name='mbs_embedding')(mbs_input)
    if bidirectional:
        x1 = Bidirectional(LSTMLayer(recurrent_units, return_sequences=True),
                           name='mbs_lstm')(e)
    else:
        x1 = LSTMLayer(recurrent_units, return_sequences=True,
                       name='mbs_lstm')(e)

    # Output
    x = GlobalAveragePooling1D(name='pooling')(x1)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='linear', name='fc')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='fc_output',
                   activity_regularizer=l2(0.002))(x)

    # Define the model
    model = Model(inputs=[mbs_input],
                  outputs=[output])

    return model
