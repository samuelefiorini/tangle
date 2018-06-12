"""This module has all the functions needed to detect concessionals."""

from __future__ import division, print_function

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from mbspbs10pc.utils import flatten

__C0C1_THRESH__ = 0.75


def find_continuously_concessionals(pbs_files):
    """"Find continuously concessionals.

    Find subjects that are using concessional cards for at least 75% of the
    years of observation.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    Returns:
    --------------
    idx: set
        The set of PTNT_IDs after the filtering step.
    """
    c0c1 = []  # init as empty list

    with warnings.catch_warnings():  # ignore FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # scan the PBS files and select only the two columns of interest and save
        # the unique PTNT_ID of the subjects using C0 or C1 in the current year
        for pbs in tqdm(pbs_files, leave=False):
            df = pd.read_csv(pbs, header=0,
                             usecols=['PTNT_ID', 'PTNT_CTGRY_DRVD_CD'])
            _c0c1 = df.loc[df['PTNT_CTGRY_DRVD_CD'].isin(['C0', 'C1'])]['PTNT_ID']
            c0c1.append(np.unique(_c0c1.values))
    c0c1 = flatten(c0c1)

    # then count the number of times an index appears
    c0c1_counts = pd.DataFrame.from_dict(Counter(c0c1), orient='index').rename({0: 'COUNTS'}, axis=1)
    # return only the subjects that use concessional cards for at least
    # 50% of the years of observation
    idx = c0c1_counts[c0c1_counts['COUNTS'] > __C0C1_THRESH__*len(pbs_files)].index.tolist()
    return set(idx)


def find_consistently_concessionals(pbs_files):
    """Find consistently concessionals.

    Find subjects that use their concessional cards for at least 75% of the PBS
    benefit items of each year.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    Returns:
    --------------
    idx: set
        The PTNT_IDs after the filtering step.
    """
    idx = set()  # init the output as an empty set

    with warnings.catch_warnings():  # ignore FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # scan the PBS files and select only the two columns of interest
        for pbs in tqdm(pbs_files, leave=False):
            df = pd.read_csv(pbs, header=0,
                             usecols=['PTNT_ID', 'PTNT_CTGRY_DRVD_CD'])
            # count the number of PBS items of each PTNT_ID
            df_counter = Counter(df['PTNT_ID'].values)

            # now keep only the PTNT_ID of the subjects using C0 or C1 and count
            # the number of PBS items each
            c0c1 = df.loc[df['PTNT_CTGRY_DRVD_CD'].isin(['C0', 'C1'])]
            c0c1_counter = Counter(c0c1['PTNT_ID'].values)

            # now calculate the concessional card usage ratio
            usage = {}
            for k in c0c1_counter.keys():
                usage[k] = c0c1_counter[k] / df_counter[k]
            usage_df = pd.DataFrame.from_dict(usage, orient='index').rename({0: 'COUNTS'}, axis=1)

            # and keep only the PTNT_ID that use it for at least 75% of the times
            usage_df = usage_df[usage_df['COUNTS'] >= __C0C1_THRESH__]

            # add them to the output set
            for i in usage_df.index: # FIXME find a way to avoid nested loops
                idx.add(i)

    # return all the unique identifiers PTNT_ID that consistently used their
    # concessional cards for at least one observation year
    return idx
