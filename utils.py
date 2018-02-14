"""Utility functions to load/visualize MBS-PBS 10% dataset."""

from __future__ import division

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
