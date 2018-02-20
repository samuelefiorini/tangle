"""This is simply a function module for `find_concessionals.py`."""

from __future__ import division, print_function

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

__C0C1_THRESH__ = 0.5


def flatten(x):
    """Flatten a list."""
    return [y for l in x for y in flatten(l)] \
        if type(x) in (list, np.ndarray) else [x]


def find_continuously_concessionals(pbs_files):
    """"Find continuously concessionals.

    Find subjects that are using concessional cards for at least 50% of the
    years of observation.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    Returns:
    --------------
    idx: pandas.Index
        The PTNT_IDs after the filtering step.
    """
    c0c1 = []
    # scan the pbs files and select only the two columns of interest and save
    # the PTNT_ID of the subjects using C0 or C1
    with warnings.catch_warnings(): # ignore FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for pbs in tqdm(pbs_files):
            df = pd.read_csv(pbs, header=0, index_col=0,
                             usecols=['PTNT_ID', 'PTNT_CTGRY_DRVD_CD'])
            _c0c1 = df.loc[df['PTNT_CTGRY_DRVD_CD'].isin(['C0', 'C1'])].index
            c0c1.append(np.unique(_c0c1))
    c0c1 = flatten(c0c1)
    # then count the number of times an index appears
    c0c1_counts = pd.DataFrame.from_dict(Counter(c0c1), orient='index').rename({0: 'COUNTS'}, axis=1)
    # return only the subjects that use concessional cards for at least
    # 50% of the years of observation
    return c0c1_counts[c0c1_counts['COUNTS'] >= __C0C1_THRESH__*len(pbs_files)].index
