"""Build keras models."""
from keras.layers import (LSTM, Bidirectional, Dense, Dot, Dropout,
                          Embedding, GlobalAveragePooling1D, Input, Multiply)
from keras.models import Model
from keras.regularizers import l2

from tangle.layers import TimespanGuidedNeuralAttention, NeuralAttention

__implemeted_models__ = ['baseline', 'attention', 'tangle']


def build_tangle(mbs_input_shape, timespan_input_shape, vocabulary_size,
                 embedding_size=50, recurrent_units=8, attention_units=8,
                 dense_units=16, bidirectional=True, LSTMLayer=LSTM,
                 **kwargs):
    """Build the keras tangle model.


    Parameters:
    --------------
    mbs_input_shape: list
        Shape of the MBS sequence input. This parameter should be equal to
        `(maxlen,)` where `maxlen` is the maximum number of elements in a
        sequence.

    timespan_input_shape: list
        Shape of the timespan sequence input. It should be `(maxlen, 1)`.

    vocabulary_size: int
        The size of the tokenizer vocabulary. This parameter is used to build
        the embedding matrix.

    embedding_size: int (default=50)
        The size of the used word embedding, it should be in [50, 100, 200,
        300] according to `glove.6B`.

    recurrent_units: int (default=8)
        The number of recurrent units in the LSTM layers.

    attention_units: int (default=8)
        The number of units of the hidden attention representation.

    dense_units: int (default=16)
        The number of dense units in the dense layers.

    bidirectional: bool (default=True)
        This flag control wether or bidirectional LSTM layers should be used.

    LSTMLayer: keras.layer (default=keras.layer.LSTM)
        This parameter controls which implementation of the LSTM layer should
         be used. There are two possibilities:
            - `keras.layer.LSTM`: slower implementation, that enables the use
              of recurrent dropout and the extraction of the activations by
              `tangle.read_activations.get_activations()`
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

    # Channel 2: Timespans
    timespan_input = Input(shape=timespan_input_shape,
                           name='timespan_input')
    if bidirectional:
        x2 = Bidirectional(LSTMLayer(recurrent_units, return_sequences=True),
                           name='timespan_lstm')(timespan_input)
    else:
        x2 = LSTMLayer(recurrent_units, return_sequences=True,
                       name='timespan_lstm')(timespan_input)

    # Timespan-guided neural attention weights
    alpha = TimespanGuidedNeuralAttention(attention_units,
                                          name='tangle_attention')([x1, x2])

    # Combine channels to get contribution and context
    c = Multiply(name='contribution')([alpha, x1])
    x = Dot(axes=1, name='context')([c, e])

    # Output
    x = GlobalAveragePooling1D(name='pooling')(x)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='relu', name='fc')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='fc_output',
                   activity_regularizer=l2(0.001))(x)

    # Define the model
    model = Model(inputs=[mbs_input, timespan_input],
                  outputs=[output])

    return model


def build_attention_model(mbs_input_shape, timespan_input_shape,
                          vocabulary_size, embedding_size=50,
                          recurrent_units=8, attention_units=8, dense_units=16,
                          bidirectional=True, LSTMLayer=LSTM, **kwargs):
    """Build keras attention model.

    This function builds the standard attention model, as in
    "Hierarchical Attention Networks for Document Classification"
    by Yang et al, with a single attention mechanism.

    Parameters:
    --------------
    mbs_input_shape: list
        Shape of the MBS sequence input. This parameter should be equal to
        `(maxlen,)` where `maxlen` is the maximum number of elements in a
        sequence.

    timespan_input_shape: list
        Shape of the timespan sequence input. It should be `(maxlen, 1)`.
        Unused, added for consistency only.

    vocabulary_size: int
        The size of the tokenizer vocabulary. This parameter is used to build
        the embedding matrix.

    embedding_size: int (default=50)
        The size of the used word embedding, it should be in [50, 100, 200,
        300] according to `glove.6B`.

    recurrent_units: int (default=8)
        The number of recurrent units in the LSTM layers.

    attention_units: int (default=8)
        The number of units of the hidden attention representation.

    dense_units: int (default=16)
        The number of dense units in the dense layers.

    bidirectional: bool (default=True)
        This flag control wether or bidirectional LSTM layers should be used.

    LSTMLayer: keras.layer (default=keras.layer.LSTM)
        This parameter controls which implementation of the LSTM layer should
         be used. There are two possibilities:
            - `keras.layer.LSTM`: slower implementation, that enables the use
              of recurrent dropout and the extraction of the activations by
              `tangle.read_activations.get_activations()`
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

    # Attention weights
    alpha = NeuralAttention(attention_units, name='attention')(x1)

    # Combine channels to get contribution and context
    c = Multiply(name='contribution')([alpha, x1])
    x = Dot(axes=1, name='context')([c, e])

    # Output
    x = GlobalAveragePooling1D(name='pooling')(x)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='relu', name='fc')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='fc_output',
                   activity_regularizer=l2(0.001))(x)

    # Define the model
    model = Model(inputs=[mbs_input],
                  outputs=[output])

    return model


def build_baseline_model(mbs_input_shape, timespan_input_shape,
                         vocabulary_size, embedding_size=50, recurrent_units=8,
                         dense_units=16, bidirectional=True, LSTMLayer=LSTM,
                         **kwargs):
    """Build keras baseline model.

    This function builds a baseline model made of Embedding + LSTM layers only.

    Parameters:
    --------------
    mbs_input_shape: list
        Shape of the MBS sequence input. This parameter should be equal to
        `(maxlen,)` where `maxlen` is the maximum number of elements in a
        sequence.

    timespan_input_shape: list
        Shape of the timespan sequence input. It should be `(maxlen, 1)`.
        Unused, added for consistency only.

    vocabulary_size: int
        The size of the tokenizer vocabulary. This parameter is used to build
        the embedding matrix.

    embedding_size: int (default=50)
        The size of the used word embedding, it should be in [50, 100, 200,
        300] according to `glove.6B`.

    recurrent_units: int (default=8)
        The number of recurrent units in the LSTM layers.

    attention_units: int (default=8)
        The number of units of the hidden attention representation.

    dense_units: int (default=16)
        The number of dense units in the dense layers.

    bidirectional: bool (default=True)
        This flag control wether or bidirectional LSTM layers should be used.

    LSTMLayer: keras.layer (default=keras.layer.LSTM)
        This parameter controls which implementation of the LSTM layer should
         be used. There are two possibilities:
            - `keras.layer.LSTM`: slower implementation, that enables the use
              of recurrent dropout and the extraction of the activations by
              `tangle.read_activations.get_activations()`
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

    # Output
    x = GlobalAveragePooling1D(name='pooling')(x1)
    x = Dropout(0.5)(x)
    x = Dense(dense_units, activation='relu', name='fc')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='fc_output',
                   activity_regularizer=l2(0.001))(x)

    # Define the model
    model = Model(inputs=[mbs_input],
                  outputs=[output])

    return model
