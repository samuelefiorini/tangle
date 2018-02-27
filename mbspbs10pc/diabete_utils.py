"""This module has all the functions needed to detect diabetics."""

import calendar
import datetime
import multiprocessing as mp
import os
from multiprocessing import Manager

import numpy as np
import pandas as pd
from mbspbs10pc import __path__ as home
from tqdm import tqdm


def find_positive_samples(dd, cc, target_year=2012):
    """Filter the population of interest according to the input target year.

    This function returns the `'PTNT_ID'` of the subjects that started taking
    diabetes drugs in the target year.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    cc: set
        The intersection between the output of find_continuously_concessionals()
        and find_consistently_concessionals().

    target_year: integer (default=2012)
        The target year

    Returns:
    --------------
    positive_subjects: dict
        Dictionary having target patient IDs (positive class) as keys and
        SPPLY_DT as values.
    """
    # Init the postive subjects with the full list of people taking
    # diabetes drugs in the target year
    positive_subjects = set(dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv'].keys())
    positive_subjects = positive_subjects.intersection(cc)  # keep only the concessionals

    for year in np.arange(2008, target_year)[::-1]:
        curr = set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'].keys())
        curr = curr.intersection(cc)  # keep only the concessionals
        positive_subjects = set(filter(lambda x: x not in curr, positive_subjects))

    positive_subjects = {k: dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv'][k] for k in list(positive_subjects)}

    return positive_subjects


def find_negative_samples(pbs_files, dd, cc):
    """Find the negative samples PTNT_ID.

    Negative samples are subjects that were NEVER prescribed to diabetes
    controlling drugs.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    dd: dictionary
        The output of find_diabetics().

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
    for pbs in pbs_files:  # get all the patients using diabetes drugs
        diabetic_overall |= set(dd[os.path.split(pbs)[-1]])
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


def find_diabetics(pbs_files, filter_copayments=True,
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
    _dd = pd.read_csv(os.path.join(home[0], 'data', 'drugs_used_in_diabetes.csv'), header=0)

    # Fix 6-digit notation
    dd = set()  # dd should be a set for performance reasons
    for item in _dd.values.ravel():
        if len(item) < 6:
            dd.add(str(0)+item)
        else:
            dd.add(item)

    # Load the Co-payments thresholds
    if filter_copayments:
        co_payments = pd.read_csv(os.path.join(home[0], 'data', 'co-payments_08-18.csv'),
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

        index[_pbs] = find_diabetes_drugs_users(pbs, dd,
                                                co_payment=co_payment,
                                                chunksize=chunksize,
                                                n_jobs=n_jobs)
    return index


def find_diabetes_drugs_users(filename, dd, co_payment=None,
                              chunksize=10, n_jobs=1):
    """Find the diabetes drugs user from a single PBS file.

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

    chunksize: integer
        The number of rows the PBS file should be split into.

    n_jobs: integer
        The number of processes that have asyncronous access to the input file.

    Returns:
    --------------
    diabetes_drugs_users: dictionary
        The dictionary of unique patients identifiers that were prescribed to
        dibates drugs in the input pbs file. The dictionary has PTNT_ID as index
        and SPPLY_DT as value. The SPPLY_DT corresponds to the FIRST time the
        patient is prescribed to the use of any diabetes control drug.
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

    # Collapse the results in a single dictionary
    # having PTNT_ID as index and SPPLY_DT as value
    # Progressbar
    progress = tqdm(
        total=len(results.keys()),
        position=1,  # the next line is ugly, but it is just the year of the PBS
        desc="Processing PBS-{}".format(os.path.split(filename)[-1].split('_')[-1].split('.')[0]),
    )
    diabetes_drugs_users = dict()
    for k in results.keys():
        progress.update(1)
        content = results[k]
        for ptnt_id in content.keys():
            diabetes_drugs_users[ptnt_id] = content[ptnt_id]

    return diabetes_drugs_users


def process_chunk(i, chunk, results, dd, co_payment):
    """Process chunk of data frame.

    When co_payment is not None, PBS items costing less than co_payments are
    filtered out.
    """
    if co_payment is None:
        idx = chunk['ITM_CD'].isin(dd)
    else:
        idx = np.logical_and(chunk['PTNT_CNTRBTN_AMT']+chunk['BNFT_AMT'] >= co_payment,
                             chunk['ITM_CD'].isin(dd))

    content = chunk.loc[idx, ['PTNT_ID', 'SPPLY_DT']]
    content.loc[:, 'SPPLY_DT'] = pd.to_datetime(content['SPPLY_DT'], format='%d%b%Y')

    # Prepare the output
    out = dict()  # initialize the empty output dictionary
    ptnt_ids = np.unique(content['PTNT_ID'].values.ravel())  # get the unique list of patient id
    for ptnt_id in ptnt_ids:  # and for each patient id
        tmp = content[content['PTNT_ID'] == ptnt_id]  # extract the corresponding content
        out[ptnt_id] = tmp['SPPLY_DT'].min()  # and keep only the first one

    if content.shape[0] > 0:  # save only the relevant content
        results[i] = out  # so content has 'PTNT_ID' as inted and 'SPPLY_DT' as value
