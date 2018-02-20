#!/usr/bin/env python
"""Find concessional subjects in the PBS files.

This script finds continuously concessional subjects in the PBS files (2008 -
2014). In order to be considered continuously concessional, a subject must:

    1) use the provided concessional card for at least 50% of the corresponding
       PBS benefit items for each year,
    2) satisfy the condition at point 1) for at least 50% of the years we have
       under investigation
"""

from __future__ import print_function

import argparse
import os

from mbspbs10pc.utils import check_input


def parse_arguments():
    """"Parse input arguments."""
    parser = argparse.ArgumentParser(description='MBS-PBS 10% data/labels '
                                                 'extraction.')
    parser.add_argument('-r', '--root', type=str,
                        help='Dataset root folder (default=../../data).',
                        default=None)
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
    if args.target_year not in list(range(2008, 2015)):
        raise ValueError("Diabetes drug starting year must be in [2008-2014]")
    return args


def main():
    """Main make_xy.py routine."""
    args = init_main()
    print('------------------------------------------')
    print('MBS - PBS 10% dataset utility: make_xy.py')
    print('------------------------------------------')

    print('* Root data folder: {}'.format(args.root))





################################################################################

if __name__ == '__main__':
    main()
