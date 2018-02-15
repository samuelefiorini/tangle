"""Utility functions to load/visualize MBS-PBS 10% dataset."""

from __future__ import division
import os
from collections import Counter
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check_input(root):
    """Check the input dataset."""
    yrange = range(2008, 2015)
    mbs_files = ['MBS_SAMPLE_10PCT_'+str(year)+'.csv' for year in yrange]
    pbs_files = ['PBS_SAMPLE_10PCT_'+str(year)+'.csv' for year in yrange]
    sample_pin_lookout = 'SAMPLE_PIN_LOOKUP.csv'

    print('Checking input...')
    # Check for the SAMPLE_PIN_LOOKUP.csv file (not mandatory)
    if not os.path.exists(os.path.join(root, sample_pin_lookout)):
        warnings.warn('File {} not found in folder {}'.format(sample_pin_lookout, root))

    # Check for the MBS files
    for mbs in mbs_files:
        if not os.path.exists(os.path.join(root, mbs)):
            raise IOError('File {} not found in folder {}'.format(mbs, root))

    # Check for the PBS files
    for pbs in pbs_files:
        if not os.path.exists(os.path.join(root, pbs)):
            raise IOError('File {} not found in folder {}'.format(pbs, root))
    print('ok.')



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
