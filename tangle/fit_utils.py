"""Utility functions to fit/cross_validate the model on MBS-PBS 10% dataset."""

from keras.callbacks import (EarlyStopping, History, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)


def concatenate_history(h):
    """Concatenate History objects.

    Useful for layers fine tuning.

    Parameters:
    --------------
    h: list
        List of history objects to concatenate.

    Returns:
    --------------
    history: `keras.callbacks.History`
        The resulting `History` object.
    """
    h0, h1 = h[0], h[1]
    history = History()
    history.history = {}
    history.epoch = h0.epoch + h1.epoch

    for k in h0.history.keys():
        history.history[k] = h0.history[k] + h1.history[k]

    return history


def get_callbacks(RLRP_patience=7, ES_patience=15, MC_filepath=None):
    """Get the callbacks list.

    Parameters:
    --------------
    RLRP_patience: int (default=7)
        Patience for ReduceLROnPlateau.

    ES_patience: int (default=15)
        Patience for EarlyStopping.

    MC_filepath: string
        ModelCheckpoint filepath.

    Returns:
    --------------
    callbacks: list
        Callbacks list:
        [ReduceLROnPlateau(...), EarlyStopping(...), ModelCheckpoint(...)]
    """
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5, patience=RLRP_patience,
                                   min_lr=1e-6, verbose=1),
                 EarlyStopping(monitor='val_loss', patience=ES_patience),
                 ModelCheckpoint(filepath=MC_filepath + '_weights.h5',
                                 save_best_only=True, save_weights_only=True),
                 TensorBoard(log_dir='/tmp/logs', histogram_freq=3,
                             batch_size=128, write_graph=True,
                             embeddings_freq=3,
                             embeddings_layer_names=[])]
    return callbacks
