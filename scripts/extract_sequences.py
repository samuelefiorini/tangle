#!/usr/bin/env python
"""Extract raw sequences from the MBS files.

Given a list of PTNT_ID, this script aims at extracting a sequence of MBS
services usage of each individual. A possible list may be something like:

    13546846: 'GP - 26 - S - 2 - GP - 50 - R'

where GP, S and R are some clinical examinations and the number between them is
the number of days between the two events.
"""

from __future__ import print_function

import argparse
import cPickle as pkl
import os
from datetime import datetime

from mbspbs10pc.utils import check_input


def parse_arguments():
    """"Parse input arguments."""
    parser = argparse.ArgumentParser(description='MBS-PBS 10% data/labels '
                                                 'extraction.')
    parser.add_argument('-r', '--root', type=str,
                        help='Dataset root folder (default=../../data).',
                        default=None)
    parser.add_argument('-t', '--target_year', type=int,
                        help='Diabetes drug starting year.', default=2012)
    parser.add_argument('-o', '--output', type=str,
                        help='Ouput file name root.',
                        default=None)
    parser.add_argument('-s', '--skip_input_check', action='store_false',
                        help='Skip the input check (default=False).')
    parser.add_argument('-nj', '--n_jobs', type=int,
                        help='The number of processes to use.', default=4)
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
        args.output = 'DumpFile'+str(datetime.now())

    # Check target year
    if args.target_year not in list(range(2008, 2015)):
        raise ValueError("Diabetes drug starting year must be in [2008-2014]")

    return args


def main():
    """Main find_concessionas.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: extract_sequences.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* PTNT_ID list: {}'.format(args.ptnt_id))
    print('* Output files: {}.[pkl, csv, ...]'.format(args.output))
    # print('* Number of jobs: {}'.format(args.n_jobs))
    # print('* Chunk size: {}'.format(args.chunk_size))
    print('-------------------------------------------------------------------')



################################################################################

if __name__ == '__main__':
    main()
