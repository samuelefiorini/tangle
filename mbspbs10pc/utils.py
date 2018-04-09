"""Utility functions to load/visualize/check/split MBS-PBS 10% dataset."""

from __future__ import division

import os
import warnings
from collections import Counter
from multiprocessing import cpu_count

import joblib as jl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit


def flatten(x):
    """Flatten a list."""
    return [y for l in x for y in flatten(l)] \
        if type(x) in (list, np.ndarray) else [x]


def check_input(root):
    """Check the input dataset."""
    yrange = range(2008, 2015)
    mbs_files = ['MBS_SAMPLE_10PCT_'+str(year)+'.csv' for year in yrange]
    mbs_files = [os.path.join(root, mbs) for mbs in mbs_files]
    pbs_files = ['PBS_SAMPLE_10PCT_'+str(year)+'.csv' for year in yrange]
    pbs_files = [os.path.join(root, pbs) for pbs in pbs_files]
    sample_pin_lookout = os.path.join(root, 'SAMPLE_PIN_LOOKUP.csv')

    mbs_files_set = set(mbs_files)
    pbs_files_set = set(pbs_files)

    # Check for the MBS files
    for mbs in mbs_files:
        if not os.path.exists(mbs):
            warnings.warn('File {} not found'.format(mbs))
            mbs_files_set.remove(mbs)

    # Check for the PBS files
    for pbs in pbs_files:
        if not os.path.exists(pbs):
            warnings.warn('File {} not found'.format(pbs))
            pbs_files_set.remove(pbs)

    print('* Found:\n   + {} MBS files\n   + {} PBS files'.format(len(mbs_files_set), len(pbs_files_set)))

    # Check for the SAMPLE_PIN_LOOKUP.csv file (not mandatory)
    if not os.path.exists(sample_pin_lookout):
        warnings.warn('File {} not found'.format(sample_pin_lookout))


def show_most_frequent(x, top_k=25, dpi=100, column=None, **kwargs):
    """Show the most frequent elements in a bar chart.

    Parameters
    ------------------
    x      - array-like, the elements to represent
    top_k  - plot only the top_k most represented elements
    dpi    - int, dots per inch of the figure
    kwargs - dict, arguments passed to plt.figure
    """
    if column is None: column = 'Counts'

    cc = Counter(x)
    dd = pd.DataFrame(cc.values(), index=cc.keys(), columns=[column])
    dd.sort_values(by=column, ascending=False, inplace=True)

    # Limit to top_k
    if top_k > dd.shape[0]: top_k = dd.shape[0]
    dd = dd.iloc[:top_k]
    xaxis = np.arange(top_k)

    # Make plot
    plt.figure(dpi=dpi, **kwargs)
    plt.bar(xaxis, dd.values.ravel())
    plt.xticks(xaxis, dd.index, rotation='vertical')
    plt.title(column)


def applyParallel(grouped, func):
    return pd.concat(Parallel(n_jobs=cpu_count())(delayed(func)(group) for name, group in grouped))


def load_data_labels(data_filename, labels_filename):
    """Load data and labels.

    Parameters:
    --------------
    data_filename: string
        Data filename.

    labels_filename: string
        Labels filename.

    Returns:
    --------------
    dataset: pandas.DataFrame
        Input data.
    """
    labels = pd.read_csv(labels_filename, header=0).rename(
        {'Unnamed: 0': 'PIN'}, axis=1)[['PIN', 'CLASS']].set_index('PIN')

    data_pkl = jl.load(open(data_filename, 'rb')).loc[labels.index, 'seq']
    dataset = pd.DataFrame(columns=['Seq', 'Class'], index=labels.index)
    dataset.loc[:, 'Seq'] = data_pkl
    dataset.loc[:, 'Class'] = labels['CLASS']
    for idx in dataset.index:
        _tmp = dataset.loc[idx, 'Seq'].split(' ')
        dataset.loc[idx, 'mbs_seq'] = ' '.join(_tmp[::2])
        dataset.loc[idx, 'times_seq'] = ' '.join(_tmp[1::2])
    return dataset


def tokenize(data):
    """Tokenize input data.

    Parameters:
    --------------
    data: pandas.DataFrame
        The DataFrame created by `load_data_labels`.

    Returns:
    --------------
    padded_mbs_seq: array
        Padded sequence of MBS items.

    padded_timestamp_seq: array
        Padded sequence of timestamps.

    tokenizer: keras.preprocessing.text.Tokenizer
        The tokenizer object fit on the input data.
    """
    # Tokenization
    tokenizer = Tokenizer(char_level=False, lower=False, split=' ')

    # Fit on corpus and extract tokenized sequences
    tokenizer.fit_on_texts(data['mbs_seq'])
    seq = tokenizer.texts_to_sequences(data['mbs_seq'])

    # Pad tokenized sequences
    # lengths = [len(x) for x in seq]
    # maxlen = int(np.percentile(lengths, 95))
    maxlen = 250  # the last 250 items
    padded_mbs_seq = pad_sequences(seq, maxlen=maxlen, padding='pre',
                                   truncating='pre', value=0)

    # Pad timestamps
    t_seq = [map(int, data.loc[idx, 'times_seq'].split(' '))
             for idx in data.index]
    padded_timestamp_seq = pad_sequences(t_seq, maxlen=maxlen,
                                         padding='pre', truncating='pre',
                                         value=-1)

    return padded_mbs_seq, padded_timestamp_seq, tokenizer


def train_validation_test_split(data, labels, test_size=0.4,
                                validation_size=0.1, verbose=False,
                                random_state0=None, random_state1=None):
    """Split the input dataset in three non overlapping sets.

    Parameters:
    --------------
    data: list
        A list made as follows `[padded_mbs_seq, padded_timestamp_seq]`

    labels: array
        Labels vector returned by `load_data_labels()`.

    test_size: numeric (default=0.4)
        Test set size.

    validation_size: numeric (default=0.1)
        Validation set size.

    verbose: bool
        Print verbose debug messages.

    random_state0: int, RandomState instance or None, (default=None)
        Random state of the learning/test split.

    random_state1: int, RandomState instance or None, (default=None)
        Random state of the training/validation split.

    Returns:
    --------------
    train_set: tuple
        A tuple like `(train_data, y_train)` where `train_data` is a list
        like [MBS training sequence, timestamp training sequence].

    validation_set: tuple
        Same as `train_set`, but for validation set.

    test_set: tuple
        Same as `train_set`, but for test set.
    """
    # Full dataset (force local copy)
    y = np.array(labels.values.ravel())
    X, X_t = np.array(data[0]), np.array(data[1])

    # Learn / Test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state0)
    learn_idx, test_idx = next(sss.split(X, y))

    X_learn, y_learn = X[learn_idx, :], y[learn_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    X_learn_t = X_t[learn_idx, :]
    X_test_t = X_t[test_idx, :]

    if verbose:
        print('* {} learn / {} test'.format(len(y_learn), len(y_test)))

    # Training / Validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_size,
                                 random_state=random_state1)
    train_idx, valid_idx = next(sss.split(X_learn, y_learn))

    X_train, y_train = X_learn[train_idx, :], y_learn[train_idx]
    X_valid, y_valid = X_learn[valid_idx, :], y_learn[valid_idx]

    X_train_t = X_learn_t[train_idx, :]
    X_valid_t = X_learn_t[valid_idx, :]

    if verbose:
        print('* {} training / {} validation'.format(len(y_train),
                                                     len(y_valid)))
    # Packing output
    train_data = [X_train, X_train_t.reshape(len(y_train), X.shape[1], 1)]
    train_set = (train_data, y_train)

    validation_data = [X_valid, X_valid_t.reshape(len(y_valid), X.shape[1], 1)]
    validation_set = (validation_data, y_valid)

    test_data = [X_test, X_test_t.reshape(len(y_test), X.shape[1], 1)]
    test_set = (test_data, y_test)

    return train_set, validation_set, test_set
