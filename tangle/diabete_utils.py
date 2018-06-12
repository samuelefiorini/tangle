"""This module has all the functions needed to detect diabetics."""
import os
import warnings

import pandas as pd
from mbspbs10pc import __path__ as home
from tqdm import tqdm

warnings.filterwarnings('ignore')


def find_others(dd, met_idx):
    """Find the people that changed from metformin to other drugs.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    Returns:
    --------------
    idx: list
        `'PTNT_ID'` of people on metformin + other drug.

    start_date: list
        The first metformin prescription.

    end_date: list
        The first non-metformin prescription.
    """
    ids = dd['PTNT_ID']
    idx = set(ids) - set(met_idx)

    grouped = dd.groupby(by='PTNT_ID')
    filtered = grouped.filter(lambda x: x['PTNT_ID'].values[0] in idx).groupby(by='PTNT_ID')

    # Init return items
    start_date, end_date = list(), list()

    # Build output variables
    for name, group in tqdm(filtered, desc='Finalizing', leave=False):
        start_date.append(group['SPPLY_DT'].min().strftime('%Y-%m-%d'))
        end_date.append(group['SPPLY_DT'].max().strftime('%Y-%m-%d'))

    return idx, start_date, end_date


def find_met2x(dd, min_metformin=1):
    """Find the people that changed from metformin to other drugs.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    min_metformin: int
        Minimum number of metformin ONLY initial prescriptions.

    Returns:
    --------------
    idx: list
        `'PTNT_ID'` of people on metformin + other drug.

    start_date: list
        The first metformin prescription.

    end_date: list
        The first non-metformin prescription.
    """
    # Load the metformin items
    metonly = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin_items.csv'),
                  header=0).values.ravel())
    metx = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin+x_items.csv'),
                  header=0).values.ravel())

    def condition(group):
        if len(group) <= min_metformin:
            return False
        else:
            sorted_group = group.sort_values(by='SPPLY_DT')
            head = set(sorted_group.head(min_metformin)['ITM_CD'].values) # these must all be metformin
            tail = set(sorted_group.tail(-min_metformin)['ITM_CD'].values) # there should be NO metformin in here

            cond1 = head.issubset(metonly)
            cond2 = len(tail.intersection(metonly)) == 0
            cond3 = len(tail.intersection(metx)) == 0

            return cond1 and cond2 and cond3

    grouped = dd.groupby(by='PTNT_ID')
    filtered = grouped.filter(condition).groupby(by='PTNT_ID')

    # Init return items
    idx, start_date, end_date = list(), list(), list()

    # Build output variables
    for name, group in tqdm(filtered, desc='Finalizing', leave=False):
        idx.append(name)
        sorted_group = group.sort_values(by='SPPLY_DT')
        # first date FIXME: max() added for consistency
        start_date.append(sorted_group.head(1)['SPPLY_DT'].max().strftime('%Y-%m-%d'))
        # get the non metformins
        filtered_group = sorted_group[~sorted_group['ITM_CD'].isin(metonly)]
        end_date.append(filtered_group['SPPLY_DT'].min().strftime('%Y-%m-%d'))

    return idx, start_date, end_date


def find_metx(dd, min_metformin=1):
    """"Find the people on metformin + other drug.

    This function finds the patients that were prescribed to some non-metformin
    diabetes drug after an initial metformin prescription.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    min_metformin: int
        Minimum number of metformin ONLY initial prescriptions.

    Returns:
    --------------
    idx: list
        `'PTNT_ID'` of people on metformin + other drug.

    start_date: list
        The first metformin prescription.

    end_date: list
        The first non-metformin prescription.
    """
    # Load the metformin items
    metonly = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin_items.csv'),
                  header=0).values.ravel())
    metx = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin+x_items.csv'),
                  header=0).values.ravel())

    def condition(group):
        """Filtering condition."""
        if len(group) <= min_metformin:
            return False
        else:
            sorted_group = group.sort_values(by='SPPLY_DT')
            # these must all be metformin for cond1
            head = set(sorted_group.head(min_metformin)['ITM_CD'].values)
            # there should be both in here for cond2
            tail = set(sorted_group.tail(-min_metformin)['ITM_CD'].values)
            # implementing conditions
            cond1 = head.issubset(metonly)
            cond2 = len(tail.intersection(metonly)) > 0 and len(tail.intersection(metonly)) < len(tail)
            # handling special case of met+x items introduced after 2014
            cond3 = len(tail.intersection(metx)) > 0
            return cond1 and (cond2 or cond3)

    grouped = dd.groupby(by='PTNT_ID')
    filtered = grouped.filter(condition).groupby(by='PTNT_ID')

    # Init return items
    idx, start_date, end_date = list(), list(), list()

    # Build output variables
    for name, group in tqdm(filtered, desc='Finalizing', leave=False):
        idx.append(name)
        sorted_group = group.sort_values(by='SPPLY_DT')
        # first date FIXME: max() added for consistency
        start_date.append(sorted_group.head(1)['SPPLY_DT'].max().strftime('%Y-%m-%d'))
        # get the non metformins
        filtered_group = sorted_group[~sorted_group['ITM_CD'].isin(metonly)]
        end_date.append(filtered_group['SPPLY_DT'].min().strftime('%Y-%m-%d'))

    return idx, start_date, end_date


def find_metonly(dd):
    """"Find the people on metformin only.

    Parameters:
    --------------
    dd: pandas.DataFrame
        The output of find_diabetics().


    Returns:
    --------------
    idx: list
        `'PTNT_ID'` of people on metformin ONLY.

    start_date: list
        The first metformin prescription.

    end_date: list
        The last day of the last observation year (2014).
    """
    # Load the metformin items
    metonly = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin_items.csv'),
                  header=0).values.ravel())

    # Filter the metformin only
    filtered = dd.groupby(by='PTNT_ID').filter(lambda x: set(x['ITM_CD']).issubset(metonly)).groupby(by='PTNT_ID')

    # Init return items
    idx, start_date, end_date = list(), list(), list()

    # Build output variables
    for name, group in tqdm(filtered, desc='Finalizing', leave=False):
        idx.append(name)
        start_date.append(group['SPPLY_DT'].min().strftime('%Y-%m-%d'))
        end_date.append('2014-12-31')

    return idx, start_date, end_date


def find_diabetics(pbs_files, ccc=set()):
    """Search people using diabetes drugs in input PBS files.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    ccc: set
        Set of continuoly and consistently concessional subjects.

    Returns:
    --------------
    df: pandas.DataFrame
        Table containing only the records of ccc using diabetics drugs.
    """
    # Load the drugs used in diabetes list file
    _dd = pd.read_csv(os.path.join(home[0], 'data', 'drugs_used_in_diabetes.csv'),
                      header=0)

    # Fix 6-digit notation
    dd = set()  # dd should be a set for performance reasons
    for item in _dd.values.ravel():
        if len(item) < 6:
            dd.add(str(0)+item)
        else:
            dd.add(item)

    # Find the ccc diabetics and save them in a single data frame
    columns = ['ITM_CD', 'PTNT_ID', 'SPPLY_DT']
    df = pd.DataFrame(columns=columns)
    for pbs in tqdm(pbs_files, desc='PBS files loading', leave=False):
        pbs_df = pd.read_csv(pbs, header=0, engine='c',
                             usecols=columns)
        pbs_df = pbs_df[pbs_df['PTNT_ID'].isin(ccc)]  # keep only ccc
        pbs_df = pbs_df[pbs_df['ITM_CD'].isin(dd)]  # keep only diabetics
        df = pd.concat((df, pbs_df))
    df.loc[:, 'SPPLY_DT'] = pd.to_datetime(df['SPPLY_DT'], format='%d%b%Y')

    return df
