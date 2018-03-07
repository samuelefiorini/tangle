"""This module keeps the functions for sequences extraction from MBS files."""
from __future__ import division, print_function

import datetime
import multiprocessing as mp
import os
import warnings
from multiprocessing import Manager

import numpy as np
import pandas as pd
from mbspbs10pc import __path__ as home
from mbspbs10pc.concessionals_utils import flatten
from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm

___MBS_FILES_DICT__ = dict()


def timespan_encoding(days):
    """Convert the input days in the desired timespan encoding.

    This function follows this encoding:
    --------------------------------
    Time duration        | Encoding
    --------------------------------
    [same day - 2 weeks] | 0
    (2 weeks  - 1 month] | 1
    (1 month  - 3 monts] | 2
    (3 months - 1 year]  | 3
    more than 1 year     | 4
    --------------------------------

    Parameters:
    --------------
    days: int
        The number of days between any two examinations.

    Returns:
    --------------
    enc: string
        The corresponding encoding.
    """
    if days < 0:
        raise ValueError('Unsupported negative timespans')
    elif days >= 0 and days <= 14:
        enc = 0
    elif days > 14 and days <= 30:  # using the "economic" month duration
        enc = 1
    elif days > 30 and days <= 90:  # using the "economic" month duration
        enc = 2
    elif days > 90 and days <= 360:  # using the "economic" year duration
        enc = 3
    else:
        enc = 4
    return str(enc)


def worker(i, pin_split, spply_dt_split, raw_data):
    """Patient tracking worker."""
    with warnings.catch_warnings():  # ignore SettingWithCopyWarning
        warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

        # First progress
        progress = tqdm(
            total=len(___MBS_FILES_DICT__.keys()),
            position=i,
            desc="[job {}] Pre-filtering".format(i),
        )

        # Pre-filter: keep only the elements of mbs_dd that are in the current split
        # this helps in reducing the time of the next step
        small_mbs_dd = dict()
        for k in sorted(___MBS_FILES_DICT__.keys()):
            progress.update(1)
            # keep only a subset of the full MBS data
            small_mbs_dd[k] = ___MBS_FILES_DICT__[k].loc[___MBS_FILES_DICT__[k]['PIN'].isin(pin_split)]
            # change format to the right datetime format (this is gonna be useful later)
            small_mbs_dd[k].loc[:, 'DOS'] = pd.to_datetime(small_mbs_dd[k]['DOS'], format='%d%b%Y')
            # and sort by date
            small_mbs_dd[k].sort_values(by='DOS', inplace=True)
        progress.close()

        # Second progress
        progress = tqdm(
            total=len(pin_split),
            position=i,
            desc="[job {}] Sequence extraction".format(i),
        )

        # Now track down each patient in the reduced MBS files
        # for s in tqdm(split, desc='[job {}] Sequence extraction'.format(i)):
        for i in range(len(pin_split)):
            progress.update(1)
            pin = pin_split[i]  # the current patient PIN
            date = spply_dt_split[i]  # the first supply date

            # create a temporary data frame storing only the information relevant
            # to the current pin
            tmp = pd.DataFrame(columns=['PIN', 'DOS', 'PINSTATE', 'BTOS-4D'])
            for k in sorted(___MBS_FILES_DICT__.keys()):
                tmp = pd.concat((tmp, small_mbs_dd[k].loc[small_mbs_dd[k]['PIN'] == pin, :]))

            # retrieve the first diabetes-related examination and
            # exclude the MBS items coming after that date
            if date is not None:  # check for the negative class
                tmp = tmp[tmp['DOS'] < date]

            if len(tmp['BTOS-4D'].values) > 0:
                # evaluate the first order difference and convert each entry in WEEKS
                timedeltas = map(lambda x: pd.Timedelta(x).days,
                                 tmp['DOS'].values[1:] - tmp['DOS'].values[:-1])
                # use the appropriate encoding
                timedeltas = map(timespan_encoding, timedeltas)
                # then build the sequence as ['exam', idle-days, 'exam', idle-days, ...]
                seq = flatten([[btos, dt] for btos, dt in zip(tmp['BTOS-4D'].values, timedeltas)])
                seq.append(tmp['BTOS-4D'].values[-1])  # add the last exam (ignored by zip)
                # and finally collapse everything down to a string like 'G0G1H...'
                seq = ''.join(map(str, seq))
                # compute the average age during the treatment by computing the average year
                avg_year = np.mean(pd.DatetimeIndex(tmp['DOS'].values.ravel()).year)
                # extract the last pinstate
                last_pinstate = tmp['PINSTATE'].values.ravel()[-1]
                # build up the result
                raw_data[pin] = (seq, avg_year, last_pinstate)
            else:
                raw_data[pin] = (list(), None, None)
        progress.close()


def get_raw_data(mbs_files, sample_pin_lookout, exclude_pregnancy=False, source=None, n_jobs=4):
    """Extract the sequences and find the additional features.

    This function, given the input list of patient identifiers `source` scans
    the `mbs_files` and extract the MBS items sequence. It also finds additional
    info on the patient such as age and sex from the `sample_pin_lookout` file.
    Unstructured sequences and additional info are referred to as raw_data.

    Parameters:
    --------------
    mbs_files: list
        List of input MBS files.

    sample_pin_lookout: string
        Location of the `SAMPLE_PIN_LOOKUP.csv` file.

    exclude_pregnancy: bool
        Exclude subjectes underatking pregnancy-related items.

    source: string
        Location of the 'PTNT_ID' csv file generated by `labels_assignment.py`.

    n_jobs: integer
        The number of processes that have asyncronous access to the input files.

    Returns:
    --------------
    raw_data: dictionary
        A dictionary that stores raw unstructured sequences of each input
        subject and additional info.

    extra_info: pandas.DataFrame
        Extra info of the input patients, such as age and sex.
    """
    raw_data = dict()

    # Step 0: load the source file, the btos4d file and the diabetes drugs file
    dfs = pd.read_csv(source, header=0, index_col=0)
    dfs['PTNT_ID'] = dfs.index  # FIXME: this is LEGACY CODE
    if 'SPPLY_DT' not in dfs.columns:  # fixing the bug with the negative class
        dfs['SPPLY_DT'] = None
    btos4d = pd.read_csv(os.path.join(home[0], 'data', 'btos4d.csv'), header=0,
                         usecols=['ITEM', 'BTOS-4D'])

    # check weather or not exclude pregnant subjects
    if exclude_pregnancy:
        pregnancy_items = set(pd.read_csv(os.path.join(home[0], 'data',
                                                       'pregnancy_items.csv'),
                                          header=0, usecols=['ITEM']).values.ravel())

    # Step 1: get sex and age
    df_pin_lookout = pd.read_csv(sample_pin_lookout, header=0)
    df_pin_lookout['AGE'] = datetime.datetime.now().year - df_pin_lookout['YOB']  # this is the age as of TODAY
    dfs = pd.merge(dfs, df_pin_lookout, how='left', left_on='PTNT_ID', right_on='PIN')[['PIN', 'SEX', 'AGE', 'SPPLY_DT', 'YOB']]
    dfs = dfs.set_index('PIN')  # set PIN as index (easier access below)
    # SPPLY_DT is the date of the FIRST diabetes drug supply

    # Step 2: follow each patient in the mbs files
    # at first create a very large dictionary with all the MBS files
    # (keeping only the relevant columns)
    # It is possible here to exclude pregnant subjects
    global ___MBS_FILES_DICT__
    for mbs in tqdm(mbs_files, desc='MBS files loading'):
        dd = pd.read_csv(mbs, header=0, usecols=['PIN', 'ITEM', 'DOS', 'PINSTATE'], engine='c')
        if exclude_pregnancy: dd = dd.loc[~dd['ITEM'].isin(pregnancy_items), :]
        ___MBS_FILES_DICT__[mbs] = pd.merge(dd, btos4d, how='left', on='ITEM')

    # This large dictionary is shared across multiple processes
    manager = Manager()
    shared_raw_data = manager.dict(raw_data)
    pool = mp.Pool(n_jobs)

    # Split the patietns in n_jobs approximately equal chunks
    pin_splits = np.array_split(dfs.index, n_jobs)  # PIN is the index
    spply_dt_splits = np.array_split(dfs['SPPLY_DT'].values.ravel(), n_jobs)

    # Submit the patient tracking jobs
    results = [pool.apply_async(worker, (i, pin_splits[i], spply_dt_splits[i], shared_raw_data)) for i in range(len(pin_splits))]

    # And collect the results
    results = [p.get() for p in results]

    # Jump one line
    print('\n')

    # Break-down the raw_data dictionary in its components, i.e.: sequence, avg year
    output = dict()
    for k in tqdm(shared_raw_data.keys(), desc='Finalizing', mininterval=5):
        if shared_raw_data[k][1] is not None:
            output[k] = shared_raw_data[k][0]  # save the sequence
            dfs.loc[k, 'AVG_AGE'] = shared_raw_data[k][1] - dfs.loc[k, 'YOB']  # save the average age
            dfs.loc[k, 'PINSTATE'] = shared_raw_data[k][2]  # save the last pinstate
    dfs = dfs.dropna()  # get rid of the extra info for the empty sequences

    # Jump one line
    print('\n')

    return output, dfs
