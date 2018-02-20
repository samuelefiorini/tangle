"""This is simply a function module for `find_concessionals.py`."""

from __future__ import division, print_function

import multiprocessing as mp
import time
from collections import Counter
from multiprocessing import Manager

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mbspbs10pc.extra import sec_to_time
from tqdm import tqdm

__C0C1_THRESH__ = 0.5
__C0C1_SET__ = set(['CO', 'C1'])


def flatten(x):
    """Flatten a list."""
    return [y for l in x for y in flatten(l)] \
        if type(x) in (list, np.ndarray) else [x]


def worker(ptnt_id_df, ptnt_id):
    # filter the items in ['C0', 'C1']
    ptnt_id_df = ptnt_id_df.loc[ptnt_id_df['PTNT_CTGRY_DRVD_CD'].isin(['CO', 'C1'])]
    n_c0c1 = ptnt_id_df.shape[0]

    # filter the items of the current ptnt_id
    n_tot = ptnt_id_df.shape[0]

    out = n_c0c1/n_tot

    # Return the corresponding ratio
    return (ptnt_id, out)


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
    idx: list
        The list of PTNT_ID after the filtering step.
    """
    c0c1 = []
    # scan the pbs files and select only the two columns of interest and save
    # the PTNT_ID of the subjects using C0 or C1
    for pbs in tqdm(pbs_files):
        df = pd.read_csv(pbs, header=0, index_col=0,
                         usecols=['PTNT_ID', 'PTNT_CTGRY_DRVD_CD'])
        c0c1.append(df.loc[df['PTNT_CTGRY_DRVD_CD'].isin(['C0', 'C1'])].index.tolist())
    c0c1 = flatten(c0c1)
    # then count the number of times an index appears
    c0c1_counts = pd.DataFrame.from_dict(Counter(c0c1), orient='index').rename({1: 'COUNTS'}, axis=1)
    # return only the subjects that use concessional cards for at least
    # 50% of the years of observation
    idx = c0c1_counts[c0c1_counts['COUNTS'] >= __C0C1_THRESH__*len(pbs_files)].index
    return idx.tolist()


def __step_1(pbs_files):
    """Find, for each year, the concessionals.

    Concessional subjects have `'PTNT_CTGRY_DRVD_CD'` in ['C0', 'C1'] for at
    least 50% of the times.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.
    """
    # Scan the pbs files
    for pbs in pbs_files:
        tic = time.time()
        print(pbs) # beware PTNT_ID is in the index
        df = pd.read_csv(pbs, header=0, index_col=0,
                         usecols=['PTNT_ID','PTNT_CTGRY_DRVD_CD'])
        print('n items:', df.shape[0])

        c0c1 = df[df['PTNT_CTGRY_DRVD_CD'].isin(['C0', 'C1'])].index
        print('n c0c1 items:', c0c1.shape[0])

        c0c1_ids = np.unique(c0c1) # people that used the concessional card at least once
        # conc_flag_df = pd.DataFrame(index=c0c1_ids, columns=['Concessional'])
        print('unique id:', len(c0c1_ids))

        # ---- multiprocessing --- #
        n_jobs = 32
        # manager = Manager()
        # ns = manager.Namespace()
        # ns.df = df
        # results = manager.dict()
        pool = mp.Pool(n_jobs)  # Use n_jobs processes

        # Check each ptnt_id
        jobs = []
        for ptnt_id in c0c1_ids:
            ptnt_id_df = df.loc[ptnt_id]
            if len(ptnt_id_df) > 1:
                p = pool.apply_async(worker, [ptnt_id_df, ptnt_id])
                jobs.append(p)
        print('jobs submitted', len(jobs))

        # Collect jobs
        # results = np.array([p.get() for p in jobs])
        results = []
        for p in tqdm(jobs):
            results.append(p.get())
        results = np.array(results)

        print(results.shape)


        # # Serial for debugging
        # results = []
        # for ptnt_id in tqdm(c0c1_ids):
        #     # tic = time.time()
        #     ptnt_id_df = df.loc[ptnt_id]
        #     # toc = time.time()
        #     # print(sec_to_time(toc - tic))
        #     if len(ptnt_id_df)>1: _, out = worker(ptnt_id, ptnt_id_df)
        # results = np.array(results)



        # conc_flag_df = pd.DataFrame.from_dict(results, orient='index').rename({0: 'Concessional'}, axis=1)
        conc_flag_df = pd.DataFrame(index=results[:,0], data=results[:,1], columns=['Concessional_ratio'])
        print(conc_flag_df.head())


        print('first step: ', conc_flag_df.shape[0])
        toc = time.time()
        print(sec_to_time(toc - tic))

        break
