"""This module has all the functions needed to detect diabetics."""

import multiprocessing as mp
import os
import warnings
from multiprocessing import Manager

import numpy as np
import pandas as pd
from mbspbs10pc import __path__ as home
from tqdm import tqdm

___PBS_FILES_DICT__ = dict()
warnings.filterwarnings('ignore')


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
    _dd = dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv']
    # Init the postive subjects with the full list of people taking
    # diabetes drugs in the target year
    positive_subjects = set(_dd.keys())
    positive_subjects = positive_subjects.intersection(cc)  # keep only the concessionals

    for year in np.arange(2008, target_year)[::-1]:
        curr = set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'].keys())
        curr = curr.intersection(cc)  # keep only the concessionals
        positive_subjects = set(filter(lambda x: x not in curr, positive_subjects))

    # Retrieve the list of positive subjects
    positive_subjects = {k: _dd[k]['SPPLY_DT'].min() for k in list(positive_subjects)}

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
        curr = set(pd.read_csv(pbs, header=0, usecols=['PTNT_ID']).values.ravel())
        curr = curr.intersection(cc)  # keep only the concessionals
        # iteratively increase the set of indexes
        negative_subjects |= set(filter(lambda x: x not in diabetic_overall, curr))

    return list(negative_subjects)


def find_diabetics(pbs_files, filter_copayments=False, metformin=False,
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

    metformin: bool,
        When True, find the two additional labels: MET_ONLY and MET_AFTER.

    chunksize: integer, [DEPRECATED]
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
    _dd = pd.read_csv(os.path.join(home[0], 'data', 'drugs_used_in_diabetes.csv'),
                      header=0)

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

    # Load the metformin PBS items
    if metformin:
        met_items = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin_items.csv'),
                                    header=0).values.ravel())
    else:
        met_items = None

    # Load the PBS data in a global variable
    # but keep only the rows relevant with diabete
    global ___PBS_FILES_DICT__
    for pbs in tqdm(pbs_files, desc='PBS files loading'):
        pbs_dd = pd.read_csv(pbs, header=0, engine='c',
                             usecols=['ITM_CD', 'PTNT_ID', 'SPPLY_DT',
                                      'PTNT_CNTRBTN_AMT', 'BNFT_AMT'])
        ___PBS_FILES_DICT__[pbs] = pbs_dd[pbs_dd['ITM_CD'].isin(dd)]

    # Itereate on the pbs files and get the index of the individuals that
    # were prescribed to diabes drugs
    pbs_progress = tqdm(total=len(pbs_files),
                        position=1,
                        desc="PBS files processing",
                        leave=True)

    index = dict()
    for pbs in sorted(pbs_files):
        _pbs = os.path.split(pbs)[-1]  # more visually appealing

        if filter_copayments:  # Select the appropriate co-payment threshold
            year = int(_pbs.split('_')[-1].split('.')[0])
            co_payment = co_payments.loc[year]['GBC']
        else:
            co_payment = None

        index[_pbs] = find_diabetes_drugs_users(pbs,
                                                co_payment=co_payment,
                                                met_items=met_items,
                                                chunksize=chunksize,
                                                n_jobs=n_jobs)
        pbs_progress.update(1)
    
    return index


def find_diabetes_drugs_users(pbs, co_payment=None, met_items=None,
                              chunksize=10, n_jobs=1):
    """Find the diabetes drugs user from a single PBS file.

    This function supports parallel asyncronous access to chunks of the input
    file.

    Parameters:
    --------------
    pbs: string
        PBS file name.

    co_payment: numeric (default=None)
        The Co-payment threshold of the current year.
        Source: [http://www.pbs.gov.au/info/healthpro/explanatory-notes/front/fee]

    met_items: set,
        A set containing the PBS items related to metformin.

    chunksize: integer, [DEPRECATED]
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

    # Get the list of UNIQUE patient id
    ptnt_ids = np.unique(___PBS_FILES_DICT__[pbs]['PTNT_ID'].values.ravel())
    pin_splits = np.array_split(ptnt_ids, n_jobs)  # PTNT_ID splits

    # Submit async jobs
    jobs = [pool.apply_async(worker, (i, pbs, pin_splits[i], results, co_payment)) for i in range(len(pin_splits))]

    # Collect jobs
    jobs = [p.get() for p in jobs]

    return results


def worker(i, pbs, pin_split, results, co_payment):
    """Load the info of a given subject id.

    When co_payment is not None, PBS items costing less than co_payments are
    filtered out.
    """
    progress = tqdm(
        total=len(pin_split),
        position=i+1,
        desc="Processing split-{}".format(i),
        leave=False
    )

    # Select only the items of the given user
    curr_pbs = ___PBS_FILES_DICT__[pbs]

    for k, pin in enumerate(pin_split):
        if k % 5 == 0: progress.update(5)  # update each 100 iter

        # Get the current pin
        chunk = curr_pbs.loc[curr_pbs['PTNT_ID'] == pin, :]

        # Filter for co-payment if needed
        if co_payment is not None:
            chunk = chunk[chunk['PTNT_CNTRBTN_AMT']+chunk['BNFT_AMT'] >= co_payment]

        # If the current patient is actually diabetic
        if len(chunk) > 0:
            chunk.loc[:, 'SPPLY_DT'] = pd.to_datetime(chunk['SPPLY_DT'], format='%d%b%Y')

            # so result has 'PTNT_ID' as index and ['SPPLY_DT', 'ITM_CD'] as value
            results[pin] = chunk[['SPPLY_DT', 'ITM_CD']]
