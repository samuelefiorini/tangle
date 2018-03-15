"""Utility functions to load/visualize/check MBS-PBS 10% dataset."""

from __future__ import division

import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
