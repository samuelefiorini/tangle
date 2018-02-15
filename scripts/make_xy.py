#!/usr/bin/env python
"""Script to generate the supervised dataset D = (X, y).

Our goal here is to generate a the dataset D = (X, y) from the raw MBS-PBS 10%
dataset. We will use the PBS dataset to identify which subjects after some point
in time (in between 2008-2014) were prescribed to the use of some
glycaemia-control drugs (see ../data/drugs_used_in_diabetes.csv).
These individuals will be labeled as our positive class (y = 1).

STEPS:
1. For each year filter the PTNT_ID that were prescribed of a drug listed in `data/drugs_used_in_diabetes.csv`
"""

import argparse
import cPickle as pkl
import multiprocessing as mp
import os
import pandas as pd

from multiprocessing import Manager
from mbspbs10pc.utils import check_input
from mbspbs10pc.extra import timed


def parse_arguments():
    """"Parse input arguments."""
    parser = argparse.ArgumentParser(description='MBS-PBS 10% data/labels '
                                                 'extraction.')
    parser.add_argument('-f', '--from_year', type=int,
                        help='Diabetes drug starting year.', default=2012)
    parser.add_argument('-r', '--root', type=str,
                        help='Dataset root folder (default=../../data).',
                        default=None)
    parser.add_argument('-s', '--skip_input_check', action='store_false',
                        help='Skipt the input check (default=True).')
    args = parser.parse_args()
    return args


def init_main():
    """Initialize the main routine."""
    args = parse_arguments()

    # Check input dataset
    if args.root is None:
        args.root = os.path.join('..', '..', 'data')
    if not args.skip_input_check: check_input(args.root)

    # Check starting year
    start_year = args.from_year
    if start_year not in range(2008, 2015):
        raise ValueError("Diabetes drug starting year must be in [2008-2014]")
    return args


def process_chunk(i, chunk, results, dd):
    """Process chunk of data frame."""
    ptnt_id = chunk.loc[chunk['ITM_CD'].isin(dd)]['PTNT_ID']
    if len(ptnt_id) > 0:  # save only the relevant results
        results[i] = ptnt_id.values


@timed
def find_diabetes_drugs_users(filename, dd, chunksize=10, n_jobs=1):
    """Find the diabetes drugs user from a PBS file.

    This function supports parallel asyncronous access to chunks of the input
    file.

    Parameters:
    --------------
    filename: string
        PBS file name.

    dd: pandas.Series
        Table of drugs used in diabetes.

    chunksize: integer
        The number of rows the PBS file should be split into.

    n_jobs: integer
        The number of processes that have asyncronous access to the input file.

    Returns:
    --------------
    index: set
        The set of unique patients identifiers that were prescribed to dibates
        drugs in the input pbs file
    """
    manager = Manager()
    results = manager.dict()
    pool = mp.Pool(n_jobs)  # Use n_jobs processes

    reader = pd.read_csv(filename, chunksize=chunksize)

    # Submit async jobs
    jobs = []
    for i, chunk in enumerate(reader):
        # process each data frame
        f = pool.apply_async(process_chunk, [i, chunk, results, dd])
        jobs.append(f)

    # Collect jobs
    for f in jobs:
        f.get()

    # Collapse the results in a single DataFrame
    index = set()
    for k in results.keys():
        index.add(k)

    return index


def find_population_of_interest(pbs_files, chunksize=10, n_jobs=1):
    """Search people using diabetes drugs in input PBS files.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

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

    # Itereate on the pbs files and get the index of the individuals that
    # were prescribed to diabes drugs
    pbs_years = [s.split('_')[-1].split('.')[0] for s in pbs_files]
    index = {k: None for k in pbs_years}  # init the index dictionary
    for pbs in pbs_files:
        print('Reading {} ...'.format(pbs))
        index[pbs] = find_diabetes_drugs_users(pbs, dd, chunksize=chunksize,
                                               n_jobs=n_jobs)
        print('done.')
    return index


def main():
    """Main make_xy.py routine."""
    args = init_main()

    # MBS-PBS 10% dataset files
    #mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    #sample_pin_lookout = filter(lambda x: x.startswith('SAMPLE'), os.listdir(args.root))[0]

    # Assign the labels
    pbs_files_fullpath = [os.path.join(args.root, '{}'.format(pbs)) for pbs in pbs_files]
    dfy = find_population_of_interest(pbs_files_fullpath, chunksize=5000, n_jobs=16)

    with open('../tmp/.pkl', 'wb') as f:
        pkl.dump(dfy, f)


################################################################################

if __name__ == '__main__':
    main()
