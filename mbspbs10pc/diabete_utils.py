"""This module has all the functions needed to detect diabetics."""
import multiprocessing as mp
import os
import warnings

import numpy as np
import pandas as pd
from mbspbs10pc import __path__ as home
from tqdm import tqdm

warnings.filterwarnings('ignore')


def find_metafter(dd, pos_id, target_year=2012):
    """"Find the people on metformin + other drug.

    This function finds the patients that were prescribed to some non-metformin
    diabetes drug after an initial metformin prescription.

    Remark: There is no difference between someone that changes metformin for a
    new drug, and someone who end up using both the drugs.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    pos_id: pd.DataFrame
        The DataFrame generated from the output of find_positive_samples().

    target_year: integer (default=2012)
        The target year

    Returns:
    --------------
    out: dictionary
        Dictionary having target patient IDs (metformin + other drug) as keys
        and SPPLY_DT as values.
    """
    pass


def worker(i, split, dd, metonly, D):
    for ptnt_id in tqdm(split, leave=False, desc='Split [{}]'.format(i), position=i):
        chunk = dd[dd['PTNT_ID'] == ptnt_id]
        items = set(chunk['ITM_CD'].values.tolist())
        D[ptnt_id] = chunk['SPPLY_DT'].min() if items.issubset(metonly) else None


def find_metonly(dd, n_jobs=1):
    """"Find the people on metformin only.

    Parameters:
    --------------
    dd: pandas.DataFrame
        The output of find_diabetics().

    n_jobs: int
        The number of parallel jobs to run, default = 1.

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

    # Parallel processing stuff
    pool = mp.Pool(n_jobs)
    manager = mp.Manager()
    D = manager.dict()
    ptnt_id_splits = np.array_split(dd['PTNT_ID'].unique(), n_jobs)

    # Submit and collect
    jobs = [pool.apply_async(worker, (i, split, dd, metonly, D)) for i, split in enumerate(ptnt_id_splits)]
    jobs = [p.get() for p in jobs]

    # Init return items
    idx, start_date, end_date = list(), list(), list()

    # Build output variables
    results = dict(D)
    for k in tqdm(results.keys(), desc='Finalizing', leave=False):
        if results[k] is not None:
            idx.append(k)
            start_date.append(results[k])
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
