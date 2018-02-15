#!/usr/bin/env python
"""Script to generate the supervised dataset D = (X, y).

Our goal here is to generate a the dataset D = (X, y) from the raw MBS-PBS 10%
dataset. We will use the PBS dataset to identify which subjects after some point
in time (in between 2008-2014) were prescribed to the use of some
glycaemia-control drugs (see ../data/drugs_used_in_diabetes.csv).
These individuals will be labeled as our positive class (y = 1).

STEPS:
1.
"""

import argparse
import os

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


def main():
    """Main make_xy.py routine."""
    args = parse_arguments()

    # Check input dataset
    if args.root is None:
        ROOT = os.path.join('..', '..', 'data')
    else:
        ROOT = args.root
    if not args.skip_input_check: check_input(ROOT)

    # Check starting year
    start_year = args.from_year
    if start_year not in range(2008, 2015):
        raise ValueError("Diabetes drug starting year must be in [2008-2014]")

################################################################################

if __name__ == '__main__':
    main()
