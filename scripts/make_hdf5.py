#!/usr/bin/env python
"""Load the MBS-PBS dataset and sotre it in an HDF5 archive."""

from __future__ import print_function

import argparse
import datetime
import os
import warnings

import pandas as pd
from mbspbs10pc.utils import check_input
from tqdm import tqdm


def parse_arguments():
    """"Parse input arguments."""
    parser = argparse.ArgumentParser(description='MBS-PBS 10% data/labels '
                                                 'extraction.')
    parser.add_argument('-r', '--root', type=str,
                        help='Dataset root folder (default=../../data).',
                        default=None)
    parser.add_argument('-o', '--output', type=str,
                        help='Ouput file name root.',
                        default=None)
    parser.add_argument('-sic', '--skip_input_check', action='store_false',
                        help='Skip the input check (default=False).')
    parser.add_argument('-cs', '--chunk_size', type=int,
                        help='The numer of rows each process has access to.',
                        default=1000)
    args = parser.parse_args()
    return args


def init_main():
    """Initialize the main routine."""
    args = parse_arguments()

    # Check input dataset
    if args.root is None:
        args.root = os.path.join('..', '..', 'data')
    if args.skip_input_check: check_input(args.root)

    # Check output filename
    if args.output is None:
        args.output = 'HDFStore_'+str(datetime.now())

    return args


def csv2hdf(csv_files, hdf_filename, chunksize=1000, tag=''):
    """Concatenate the input csv_files and save them in a single HDF5 store.

    Parameters:
    --------------
    csv_files: list
        List of MBS-PBS csv files.

    hdf_filename: string
        Name of the generated output file
    """
    # Init the empty output table
    store = pd.HDFStore(hdf_filename)

    # Concat everything in a single data frame
    for csv in tqdm(csv_files, desc='Loading {}'.format(tag)):
        for chunk in pd.read_csv(csv, header=0, index_col=0, chunksize=chunksize):
            store.append(os.path.split(csv)[-1].split('.')[0], chunk)

    store.close()


def main():
    """Main make_hdf5.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: make_hdf5.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* Output files: {}_*.hdf5'.format(args.output))
    print('* Chunk size: {}'.format(args.chunk_size))
    print('-------------------------------------------------------------------')

    # MBS 10% dataset files
    mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    mbs_files_fullpath = [os.path.join(args.root, mbs) for mbs in mbs_files]

    # Create the MBS HDF5 store
    filename = args.output+'_MBS.h5'
    csv2hdf(mbs_files_fullpath, hdf_filename=filename, tag='MBS')

    # PBS 10% dataset files
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    pbs_files_fullpath = [os.path.join(args.root, pbs) for pbs in pbs_files]

    # Create the PBS HDF5 store
    filename = args.output+'_PBS.h5'
    csv2hdf(pbs_files_fullpath, hdf_filename=filename, tag='PBS')


################################################################################

if __name__ == '__main__':
    with warnings.catch_warnings():  # ignore FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        main()
