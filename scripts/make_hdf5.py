#!/usr/bin/env python
"""Load the MBS-PBS dataset and sotre it in an HDF5 archive."""

from __future__ import print_function

import argparse
import datetime
import os

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


def csv2hdf(csv_files, hdf_filename):
    """Concatenate the input csv_files and save them in a single HDF5 store.

    Parameters:
    --------------
    csv_files: list
        List of MBS-PBS csv files.

    hdf_filename: string
        Name of the generated output file
    """
    # Init the empty output table
    cols = pd.read_csv(csv_files[0], nrows=1, header=0, index_col=0).columns
    cols = cols.append(pd.Index(['SOURCE']))
    df = pd.DataFrame(columns=cols)

    # Concat everything in a single data frame
    for csv in tqdm(csv_files):
        tmp = pd.read_csv(csv, header=0, index_col=0)
        tmp['SOURCE'] = os.path.split(csv)[-1]
        df = pd.concat((df, tmp))
        del tmp  # easy garbage collection

    print('* Saving {} '.format(hdf_filename), end=' ')
    df.to_hdf(hdf_filename, 'data', mode='w', format='fixed')
    print(u'\u2713')


def main():
    """Main make_hdf5.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: make_hdf5.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* Output files: {}.hdf5'.format(args.output))
    print('-------------------------------------------------------------------')

    # MBS 10% dataset files
    mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    mbs_files_fullpath = [os.path.join(args.root, mbs) for mbs in mbs_files]

    # Create the MBS HDF5 store
    filename = args.output+'_MBS_.h5'
    csv2hdf(mbs_files_fullpath, hdf_filename=filename)

    # PBS 10% dataset files
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    pbs_files_fullpath = [os.path.join(args.root, pbs) for pbs in pbs_files]

    # Create the PBS HDF5 store
    filename = args.output+'_PBS_.h5'
    csv2hdf(pbs_files_fullpath, hdf_filename=filename)


################################################################################

if __name__ == '__main__':
    main()
