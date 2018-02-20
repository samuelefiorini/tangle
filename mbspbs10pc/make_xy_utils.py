"""This is simply a function module for `make_xy.py`."""

import calendar
import datetime
import multiprocessing as mp
import os
from multiprocessing import Manager

import numpy as np
import pandas as pd
from tqdm import tqdm
# from mbspbs10pc.extra import timed


def process_chunk(i, chunk, results, dd, co_payment):
    """Process chunk of data frame.

    When co_payment is not None, PBS items costing less than co_payments are
    filtered out.
    """
    if co_payment is None:
        idx = chunk['ITM_CD'].isin(dd)
    else:
        idx = np.logical_and(chunk['PTNT_CNTRBTN_AMT']+chunk['BNFT_AMT']>=co_payment,
                             chunk['ITM_CD'].isin(dd))

    content = chunk.loc[idx][['PTNT_ID', 'SPPLY_DT']]

    if content.shape[0] > 0:  # save only the relevant content
        results[i] = content  # contenta has 'PTNT_ID' and 'SPPLY_DT'


# @timed
def find_diabetes_drugs_users(filename, dd, co_payment=None,
                              monthly_breakdown=False, chunksize=10, n_jobs=1):
    """Find the diabetes drugs user from a PBS file.

    This function supports parallel asyncronous access to chunks of the input
    file.

    Parameters:
    --------------
    filename: string
        PBS file name.

    dd: pandas.Series
        Table of drugs used in diabetes.

    co_payment: numeric (default=None)
        The Co-payment threshold of the current year.
        Source: [http://www.pbs.gov.au/info/healthpro/explanatory-notes/front/fee]

    monthly_breakdown: bool (default=False)
        When True, split the records in different months for each year.

    chunksize: integer
        The number of rows the PBS file should be split into.

    n_jobs: integer
        The number of processes that have asyncronous access to the input file.

    Returns:
    --------------
    index: list or dictionary
        The list of unique patients identifiers that were prescribed to dibates
        drugs in the input pbs file. If monthly_breakdown is True, the function
        returns a dictionary of lists, where each key is a month.
        E.g.: {1: (232,2312,442,...), 2: (11,678,009,...), ...}
    """
    manager = Manager()
    results = manager.dict()
    pool = mp.Pool(n_jobs)  # Use n_jobs processes

    reader = pd.read_csv(filename, chunksize=chunksize,
                         usecols=['ITM_CD', 'PTNT_ID', 'SPPLY_DT',
                                  'PTNT_CNTRBTN_AMT', 'BNFT_AMT'])
    # Submit async jobs
    jobs = []
    for i, chunk in enumerate(reader):
        # process each data frame
        f = pool.apply_async(process_chunk, [i, chunk, results, dd, co_payment])
        jobs.append(f)

    # Collect jobs
    for f in jobs:
        f.get()

    # Check for monthly breakdown flag
    if monthly_breakdown:
        indexes = {m: set() for m in range(1, 13)}  # each key is a month
        year = int(results[0]['SPPLY_DT'].values[0][-4:])  # retrieve the current year
        #_full_index = set() # avoid duplicates in the same year

        for k in results.keys(): # collapse all the results in a single object
            content = results[k]
            content['SPPLY_DT'] = pd.to_datetime(content['SPPLY_DT'], format='%d%b%Y') # set the right date format

            for month in range(1, 13):  # search for all possible months
                _, last_day = calendar.monthrange(year, month)

                # filter the items of the current month
                ptnt_id_month = content[np.logical_and(content['SPPLY_DT'] >= datetime.date(year=year, month=month, day=1), content['SPPLY_DT'] <= datetime.date(year=year, month=month, day=last_day))]['PTNT_ID']

                for item in ptnt_id_month:  # iterate over the possible PTNT_IDs
                    indexes[month].add(item)
                    #if item not in _full_index: indexes[month].add(item) # avoid duplicates in the same year
                    #_full_index.add(item)

        return {m: list(indexes[m]) for m in range(1, 13)}

    else:
        # Collapse the results in a single set
        index = set()
        for k in results.keys():
            content = results[k]['PTNT_ID']  # extrapolate the only relevant field
            for item in content:  # FIXME find a way to avoid nested loops
                index.add(item)

        return list(index)


# @timed
def find_population_of_interest(pbs_files, filter_copayments=True, monthly_breakdown=False,
                                chunksize=10, n_jobs=1):
    """Search people using diabetes drugs in input PBS files.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    filter_copayments: bool
        When True, entries in the PBS data having
        total cost < copayment(year) will be excluded. This improves
        consistency for entries that are issued after April 2012. This only
        holds for General Beneficiaries.

    monthly_breakdown: bool (default=False)
        When True, split the records in different months for each year.

    chunksize: integer
        The number of rows the PBS file should be split into.

    n_jobs: integer
        The number of processes that have asyncronous access to the input file.

    Returns:
    --------------
    index: dictionary
        Dictionary having PBS filenames as keys and the corresponding
        `'PTNT_ID'`s of individuals prescribed to diabete control drug as
        values.

        E.g.: `{'PBS_SAMPLE_10PCT_2012.csv': [3928691704, 5156241855,...]}`
    """
    # Load the drugs used in diabetes list file
    _dd = pd.read_csv(os.path.join('data', 'drugs_used_in_diabetes.csv'), header=0)

    # Fix 6-digit notation
    dd = set()  # dd should be a set for performance reasons
    for item in _dd.values.ravel():
        if len(item) < 6:
            dd.add(str(0)+item)
        else:
            dd.add(item)

    # Load the Co-payments thresholds
    if filter_copayments:
        co_payments = pd.read_csv(os.path.join('data', 'co-payments_08-18.csv'),
                                  header=0, index_col=0, usecols=['DOC', 'GBC'])

    # Itereate on the pbs files and get the index of the individuals that
    # were prescribed to diabes drugs
    index = dict()
    for pbs in tqdm(pbs_files):
        _pbs = os.path.split(pbs)[-1]  # more visually appealing

        if filter_copayments:  # Select the appropriate co-payment threshold
            year = int(_pbs.split('_')[-1].split('.')[0])
            co_payment = co_payments.loc[year]['GBC']
        else:
            co_payment = None

        # print('- Reading {} ...'.format(_pbs))
        index[_pbs] = find_diabetes_drugs_users(pbs, dd,
                                                co_payment=co_payment,
                                                monthly_breakdown=monthly_breakdown,
                                                chunksize=chunksize,
                                                n_jobs=n_jobs)
        # print('done.')
    return index


def find_positive_samples(dd, target_year=2012):
    """Filter the population of interest according to the input target year.

    This function returns the `'PTNT_ID'` of the subjects that started taking
    diabetes drugs in the target year.

    Parameters:
    --------------
    dd: dictionary
        The output of find_population_of_interest().

    target_year: integer (default=2012)
        The target year

    Returns:
    --------------
    positive_subjects: list
        The list of target patient IDs (positive class).
    """
    if isinstance(dd[dd.keys()[0]], dict):
        # -- Month-by-month analysis -- #
        raise NotImplementedError('Monthly breakdown not yet implemented.')
    else:
        # -- Year-by-year analysis -- #
        # Init the postive subjects with the full list of people taking
        # diabetes drugs in the target year
        positive_subjects = set(dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv'])

        for year in np.arange(2008, target_year)[::-1]:
            curr = set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'])
            positive_subjects = set(filter(lambda x: x not in curr, positive_subjects))

        return list(positive_subjects)

# @timed
def find_negative_samples(pbs_files, dd):
    """Find the negative samples PTNT_ID.

    Negative samples are subjects that were NEVER prescribed to diabetes
    controlling drugs.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    dd: dictionary
        The output of find_population_of_interest().

    Returns:
    --------------
    negative_subjects: list
        The list of non target patient IDs (negative class).
    """
    if isinstance(dd[dd.keys()[0]], dict):
        # -- Month-by-month analysis -- #
        raise NotImplementedError('Monthly breakdown not yet implemented.')
    else:
        # -- Year-by-year analysis -- #
        # Start with an empty set, the iterate on each year and iteratively add
        # to the negative subjects list, the patient id that are not selected
        # as diabetic at the previous step.
        negative_subjects = set()

        diabetic_overall = set()
        for year in np.arange(2008, 2014): # FIXME as soon as you get all PBS files
            diabetic_overall |= set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'])

        for pbs in tqdm(pbs_files): # TODO maybe use multiprocessing here
            _pbs = os.path.split(pbs)[-1]  # more visually appealing

            # print('- Reading {} ...'.format(_pbs))
            curr = set(pd.read_csv(pbs, header=0, usecols=['PTNT_ID']).values.ravel())
            # iteratively increase the set of indexes
            negative_subjects |= set(filter(lambda x: x not in diabetic_overall, curr))
            # print('done.')

        return list(negative_subjects)


def extract_sequences(mbs_files, pntn_id):
    """Extract the raw sequences from the MBS files for each input subject.

    Parameters:
    --------------
    mbs_files: list
        List of input MBS filenames.

    ptnt_id: list:
        The list of patient IDs returned find_<XXX>_samples(), where <XXX> is
        either 'positive' or 'negative'.
    """
    return 0
