"""Bidirectional timestamp-guided attention model.

Keras implementation.
"""
from keras import backend as K
from keras.layers import (LSTM, Add, Bidirectional, Dense, Dropout, Embedding,
                          GlobalAveragePooling1D, Input, Lambda, Multiply,
                          Permute, RepeatVector)
from keras.models import Model
from keras.regularizers import l2


def build_model(mbs_input_shape, timestamp_input_shape, vocabulary_size,
                embedding_size=50, recurrent_units=8, dense_units=16,
                bidirectional=True, single_attention=False,
                LSTMLayer=LSTM):
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

    single_attention: bool(default=False)
    This flag control wether or not using a single attention vector shared
    across multiple recurrent hidden states or keep all of them. If yes, the
    attention vector is computed as the mean of all the attentions evaluated
    for each hidden state dimension.

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

    # -- Timestamp-guided attention -- #
    # Channel 2: Timestamps
    timestamp_input = Input(shape=timestamp_input_shape,
                            name='timestamp_input')
    if bidirectional:
        x2 = Bidirectional(LSTMLayer(recurrent_units, return_sequences=True),
                           name='timestamp_lstm')(timestamp_input)
    else:
        x2 = LSTMLayer(recurrent_units, return_sequences=True,
                       name='timestamp_lstm')(timestamp_input)
    x2 = Permute((2, 1), name='transpose_timestamp')(x2)
    x2 = Dense(mbs_input_shape[0], activation='linear',
               name='timestamp_dense')(x2)

    # Attention probability distribution
    alpha = Permute((2, 1), name='hidden_to_time_permute')(x1)
    alpha = Dense(mbs_input_shape[0], activation='linear',
                  name='alpha_dense')(alpha)
    alpha = Add(name='timestamp_induced_attention')([alpha, x2])

    alpha = Dense(mbs_input_shape[0], activation='tanh',
                  name='attention_tanh')(alpha)
    alpha = Dense(mbs_input_shape[0], activation='softmax',
                  name='attention_matrix')(alpha)
    if single_attention:  # obtain a single attention vector by averaging
        alpha = Lambda(lambda x: K.mean(x, axis=1),
                       name='attention_probabilities')(alpha)
        if bidirectional:
            alpha = RepeatVector(2 * recurrent_units)(alpha)
        else:
            alpha = RepeatVector(recurrent_units)(alpha)
    alpha = Permute((2, 1), name='time_to_hidden_permute')(alpha)
    # -- Timestamp-guided attention -- #

    # Combine channels to get context
    context = Multiply(name='context_creation')([alpha, x1])

    # Output
    x = GlobalAveragePooling1D(name='pooling')(context)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid',
                   activity_regularizer=l2(0.002))(x)

    # Define the model
    model = Model(inputs=[mbs_input, timestamp_input],
                  outputs=[output])
    return model
