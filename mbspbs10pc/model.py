"""Bidirectional timestamp-guided attention model.

Keras implementation.
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import (LSTM, Add, Bidirectional, Dense, Dropout, Embedding,
                          GlobalAveragePooling1D, Input, Lambda, Multiply,
                          Permute, RepeatVector)
from keras.models import Model
from keras.regularizers import l2


class TimestampGuidedAttention(Layer):
    def __init__(self, dense_units, use_bias=True, **kwargs):
        """Implementation of the Timestamp guided attention layer."""
        self.dense_units = dense_units
        self.use_bias = use_bias
        self.output_dim = None
        super(TimestampGuidedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is:
        # [(None, timesteps, hidden_units), (None, timesteps, hidden_units)]
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')
        if len(input_shape) < 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        hidden_units = [s[1] for s in input_shape if s is not None]
        if len(set(hidden_units)) > 1:
            raise ValueError('The two inputs should have the same number of '
                             'hidden units. Got {}.'.format(hidden_units))
        n_h = hidden_units[0]

        # Dense MBS-items weights (linear activation)
        self.kernel_x = self.add_weight(name='kernel_x',
                                        shape=(n_h, self.dense_units),
                                        initializer='glorot_uniform',
                                        trainable=True)
        # Dense timestamp weights (linear activation)
        self.kernel_t = self.add_weight(name='kernel_t',
                                        shape=(n_h, self.dense_units),
                                        initializer='glorot_uniform',
                                        trainable=True)
        # Dense (MBS-items * timestamp) weights (tanh activations)
        self.kernel_d = self.add_weight(name='kernel_d',
                                        shape=(n_h, self.dense_units),
                                        initializer='glorot_uniform',
                                        trainable=True)
        # Dense weights (softmax activations)
        self.kernel_a = self.add_weight(name='kernel_a',
                                        shape=(n_h, self.dense_units),
                                        initializer='glorot_uniform',
                                        trainable=True)
        if self.use_bias:
                self.bias_x = self.add_weight(name='bias_x',
                                              shape=(self.dense_units,),
                                              initializer='zeros',
                                              trainable=True)
                self.bias_t = self.add_weight(name='bias_t',
                                              shape=(self.dense_units,),
                                              initializer='zeros',
                                              trainable=True)
                self.bias_d = self.add_weight(name='bias_d',
                                              shape=(self.dense_units,),
                                              initializer='zeros',
                                              trainable=True)

        # The output dimension should be the same as the input one
        self.output_dim = input_shape[0]
        super(TimestampGuidedAttention, self).build(input_shape)

    def call(self, inputs):
        assert len(inputs) == 2
        x = Permute((2, 1))(inputs[0])  # transpose input
        t = Permute((2, 1))(inputs[1])
        # First two dense layers with linear activation
        gamma = K.dot(x, self.kernel_x)
        beta = K.dot(t, self.kernel_t)
        if self.use_bias:
            gamma = K.bias_add(gamma, self.bias_x)
            beta = K.bias_add(beta, self.bias_t)

        # Sum the two resulting tensors
        delta = Add()([gamma, beta])

        # Dense layer with tanh activation
        u = K.dot(delta, self.kernel_d)
        if self.use_bias:
            u = K.bias_add(u, self.bias_d)
        u = K.tanh(u)

        # Dense layer with softmax activation (no bias needed)
        alpha = K.softmax(K.dot(u, self.kernel_a))
        alpha = Permute((2, 1))(alpha)  # transpose back to the original shape

        return alpha

    def compute_output_shape(self, input_shape):
        return self.output_dim


def build_model(mbs_input_shape, timestamp_input_shape, vocabulary_size,
                embedding_size=50, recurrent_units=8, dense_units=16,
                bidirectional=True, LSTMLayer=LSTM):
    """Build the keras model.

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

    dense_units: int (default=16)
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
    alpha = TimestampGuidedAttention(mbs_input_shape[0],
                                     name='tsg_attention')([x1, x2])
    # -- Timestamp-guided attention -- #

    # Combine channels to get context
    context = Multiply(name='context_creation')([alpha, x1])

    # Output
    x = GlobalAveragePooling1D(name='pooling')(context)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid',
                   activity_regularizer=l2(0.002))(x)

    # Define the model
    model = Model(inputs=[mbs_input, timestamp_input],
                  outputs=[output])
    return model
