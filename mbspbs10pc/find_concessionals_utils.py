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
    idx: set
        The set of PTNT_IDs after the filtering step.
    """
    c0c1 = []  # init as empty list

    with warnings.catch_warnings():  # ignore FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # scan the PBS files and select only the two columns of interest and save
        # the PTNT_ID of the subjects using C0 or C1
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
    idx = c0c1_counts[c0c1_counts['COUNTS'] >= __C0C1_THRESH__*len(pbs_files)].index.tolist()
    return set(idx)


def find_consistently_concessionals(pbs_files):
    """Find consistently concessionals.

    Find subjects that use their concessional cards for at least 50% of the PBS
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
        for pbs in tqdm(pbs_files):
            df = pd.read_csv(pbs, header=0, index_col=0,
                             usecols=['PTNT_ID', 'PTNT_CTGRY_DRVD_CD'])
            # count the number of PBS items of each PTNT_ID
            df_counter = Counter(df.index)

            # now keep only the PTNT_ID of the subjects using C0 or C1 and count
            # the number of PBS items each
            c0c1 = df.loc[df['PTNT_CTGRY_DRVD_CD'].isin(['C0', 'C1'])]
            c0c1_counter = Counter(c0c1.index)

            # now calculate the concessional card usage ratio
            usage = {}
            for k in c0c1_counter.keys():
                usage[k] = c0c1_counter[k] / df_counter[k]
            usage_df = pd.DataFrame.from_dict(usage, orient='index').rename({0: 'COUNTS'}, axis=1)

            # and keep only the PTNT_ID that use it for at least 50% of the times
            usage_df = usage_df[usage_df['COUNTS'] >= __C0C1_THRESH__]

            # add them to the output set
            for i in usage_df.index:
                idx.add(i)
    # return all the unique identifiers PTNT_ID that consistently used their
    # concessional cards for at least one observation year
    return idx


def find_positive_samples(dd, cc, target_year=2012):
    """Filter the population of interest according to the input target year.

    This function returns the `'PTNT_ID'` of the subjects that started taking
    diabetes drugs in the target year.

    Parameters:
    --------------
    dd: dictionary
        The output of find_population_of_interest().

    cc: set
        The intersection between the output of find_continuously_concessionals()
        and find_consistently_concessionals().

    target_year: integer (default=2012)
        The target year

    Returns:
    --------------
    positive_subjects: list
        The list of target patient IDs (positive class).
    """
    # Init the postive subjects with the full list of people taking
    # diabetes drugs in the target year
    positive_subjects = set(dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv'])
    positive_subjects = positive_subjects.intersection(cc)  # keep only the concessionals

    for year in np.arange(2008, target_year)[::-1]:
        curr = set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'])
        curr = curr.intersection(cc)  # keep only the concessionals
        positive_subjects = set(filter(lambda x: x not in curr, positive_subjects))

    return list(positive_subjects)


def find_negative_samples(pbs_files, dd, cc):
    """Find the negative samples PTNT_ID.

    Negative samples are subjects that were NEVER prescribed to diabetes
    controlling drugs.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    dd: dictionary
        The output of find_population_of_interest().

    cc: set
        The intersection between the output of find_continuously_concessionals()
        and find_consistently_concessionals().

    Returns:
    --------------
    negative_subjects: list
        The list of non target patient IDs (negative class).
    """
    # Start with an empty set, the iterate on each year and iteratively add
    # to the negative subjects list, the patient id that are not selected
    # as diabetic at the previous step.
    negative_subjects = set()

    diabetic_overall = set()
    for year in np.arange(2008, 2014): # FIXME as soon as you get all PBS files
        diabetic_overall |= set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'])
    diabetic_overall = diabetic_overall.intersection(cc)   # keep only the concessionals

    for pbs in tqdm(pbs_files): # TODO maybe use multiprocessing here
        # _pbs = os.path.split(pbs)[-1]  # more visually appealing

        # print('- Reading {} ...'.format(_pbs))
        curr = set(pd.read_csv(pbs, header=0, usecols=['PTNT_ID']).values.ravel())
        curr = curr.intersection(cc)  # keep only the concessionals
        # iteratively increase the set of indexes
        negative_subjects |= set(filter(lambda x: x not in diabetic_overall, curr))
        # print('done.')

    return list(negative_subjects)
