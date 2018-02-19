#!/usr/bin/env python
"""Script to generate the supervised dataset D = (X, y).

Our goal here is to generate a the dataset D = (X, y) from the raw MBS-PBS 10%
dataset. We will use the PBS dataset to identify which subjects after some point
in time (in between 2008-2014) were prescribed to the use of some
glycaemia-control drugs (see ../data/drugs_used_in_diabetes.csv).
These individuals will be labeled as our positive class (y = 1).

* For each year filter the PTNT_ID that were prescribed of a drug listed in
  `data/drugs_used_in_diabetes.csv`

* In April 2012 they started to record every PBS item, even the ones below the
  co-payment threshold. For consistency, it is possible to exclude from the
  counts the PBS items having total cost < co-payment(year).
  Where total cost is 'BNFT_AMT'+'PTNT_CNTRBTN_AMT'.
  Be aware that the threshold varies in the years.
  See data/co-payments_08-18.csv. This only holds for General Beneficiaries.
"""

import argparse
import cPickle as pkl
import calendar
import datetime
import multiprocessing as mp
from multiprocessing import Manager
import os
import numpy as np
import pandas as pd
from mbspbs10pc.extra import timed
from mbspbs10pc.utils import check_input


def parse_arguments():
    """"Parse input arguments."""
    parser = argparse.ArgumentParser(description='MBS-PBS 10% data/labels '
                                                 'extraction.')
    parser.add_argument('-t', '--target_year', type=int,
                        help='Diabetes drug starting year.', default=2012)
    parser.add_argument('-r', '--root', type=str,
                        help='Dataset root folder (default=../../data).',
                        default=None)
    parser.add_argument('-o', '--output', type=str,
                        help='Temporary file pikle output.',
                        default=None)
    parser.add_argument('-s', '--skip_input_check', action='store_false',
                        help='Skip the input check (default=False).')
    parser.add_argument('-fc', '--filter_copayments', action='store_true',
                        help='Use this option to include the PBS '
                        'items having total cost <= co-payment(year)'
                        ' (default=True).')
    parser.add_argument('-mb', '--monthly_breakdown', action='store_true',
                        help='Split records in different months (default=True).')
    args = parser.parse_args()
    return args


def init_main():
    """Initialize the main routine."""
    args = parse_arguments()

    # Check input dataset
    if args.root is None:
        args.root = os.path.join('..', '..', 'data')
    if not args.skip_input_check: check_input(args.root)

    # Check target year
    if args.target_year not in range(2008, 2015):
        raise ValueError("Diabetes drug starting year must be in [2008-2014]")
    return args


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


@timed
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


@timed
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
    for pbs in pbs_files:
        _pbs = os.path.split(pbs)[-1]  # more visually appealing

        if filter_copayments:  # Select the appropriate co-payment threshold
            year = int(_pbs.split('_')[-1].split('.')[0])
            co_payment = co_payments.loc[year]['GBC']
        else:
            co_payment = None

        print('- Reading {} ...'.format(_pbs))
        index[_pbs] = find_diabetes_drugs_users(pbs, dd,
                                                co_payment=co_payment,
                                                monthly_breakdown=monthly_breakdown,
                                                chunksize=chunksize,
                                                n_jobs=n_jobs)
        print('done.')
    return index


@timed
def filter_population_of_interest(dd, target_year=2012):
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
    ptnt_id: list
        The list of target patient IDs.
    """
    if isinstance(dd[dd.keys()[0]], dict):
        # Monthly analysis
        raise NotImplementedError('Monthly breakdown not implemented yet.')
    else:
        # Yearly analysis
        # Init the postive subjects with the full list of people taking
        # diabetes drugs in the target year
        positive_subjects = set(dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv'])
        negative_subjects = []

        for year in np.arange(2008, target_year)[::-1]:
            curr = set(dd['PBS_SAMPLE_10PCT_'+str(year)+'.csv'])
            positive_subjects = set(filter(lambda x: x not in curr, positive_subjects))

        return list(positive_subjects), list(negative_subjects)


def main():
    """Main make_xy.py routine."""
    args = init_main()
    print('------------------------------------------')
    print('MBS - PBS 10% dataset utility: make_xy.py')
    print('------------------------------------------')

    print('[{}] Co-payment filter: {}'.format(*('+', 'ON') if args.filter_copayments else (' ', 'OFF')))
    print('[{}] Monthly breakdown: {}'.format(*('+', 'ON') if args.monthly_breakdown else (' ', 'OFF')))
    print('------------------------------------------')

    # MBS-PBS 10% dataset files
    # mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    # sample_pin_lookout = filter(lambda x: x.startswith('SAMPLE'), os.listdir(args.root))[0]

    # Filter the population of people using drugs for diabetes
    pbs_files_fullpath = [os.path.join(args.root, '{}'.format(pbs)) for pbs in pbs_files]

    # dd = find_population_of_interest(pbs_files_fullpath,
    #                                  filter_copayments=args.filter_copayments,
    #                                  monthly_breakdown=args.monthly_breakdown,
    #                                  chunksize=10000, n_jobs=32)
    #
    # # Dump results
    if args.output is None:
        filename = 'DumpFile'+str(datetime.now())+'.pkl'
    else:
        filename = args.output if args.output.endswith('.pkl') else args.output+'.pkl'

    # print('- Saving {} ...'.format(filename))
    # with open(filename, 'wb') as f:  # FIXME
    #     pkl.dump(dd, f)
    # print('done.')

    print('- Loading {} ...'.format(filename))
    with open(filename, 'rb') as f:  # FIXME
        dd = pkl.load(f)
    print('done.')

    # Find, for each year, the number of people that STARTED taking
    # drugs for diabetes; i.e.: people that are prescribed to diabetes drugs in
    # the current year and that were never prescribed before
    pos_id, neg_id = filter_population_of_interest(dd, target_year=args.target_year)

    # FIXME
    filename = filename[:-4]+'_class_1.csv'
    print('- Saving {} ...'.format(filename))
    pd.DataFrame(data=pos_id, columns=['PTNT_ID']).to_csv(filename, index=False)
    print('done.')

    filename = filename[:-4]+'_class_0.csv'
    print('- Saving {} ...'.format(filename))
    pd.DataFrame(data=neg_id, columns=['PTNT_ID']).to_csv(filename, index=False)
    print('done.')





################################################################################

if __name__ == '__main__':
    main()
