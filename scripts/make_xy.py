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
import os

import pandas as pd

from mbspbs10pc.utils import check_input


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


def main():
    """Main make_xy.py routine."""
    args = init_main()

    # MBS-PBS 10% dataset files
    mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    sample_pin_lookout = filter(lambda x: x.startswith('SAMPLE'), os.listdir(args.root))[0]

    # Load the drugs used in diabetes list file
    dd = pd.read_csv(os.path.join('data', 'drugs_used_in_diabetes.csv'), header=0)

    # Fix 6-digit notation
    dd_set = set()
    for item in dd.values.ravel():
        if len(item) < 6:
            dd_set.add(str(0)+item)
        else:
            dd_set.add(item)


################################################################################

if __name__ == '__main__':
    main()
