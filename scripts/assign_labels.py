#!/usr/bin/env python
"""Assign class labels to the population of interest.

This script will then produce a list of continuously and consistently
concessional subjects that use metformin. The following labels are assigned:

    * MET_ONLY, i.e.: patients that are using metformin ONLY
    * MET_AFTER, i.e.: patients that after a first metformin prescription
      started to use another diabetes controlling drug.
    * MET_SUB, i.e.: patients that changed from metformin to another drug
"""

from __future__ import print_function

import argparse
import os
from datetime import datetime

import joblib as jl
from mbspbs10pc import diabete_utils as d_utils
from mbspbs10pc.utils import check_input


def parse_arguments():
    """"Parse input arguments."""
    parser = argparse.ArgumentParser(description='MBS-PBS 10% data/labels '
                                                 'extraction.')
    parser.add_argument('-r', '--root', type=str,
                        help='Dataset root folder (default=../../data).',
                        default=None)
    parser.add_argument('-s', '--source', type=str,
                        help='Aux files root folder (default=./tmp).',
                        default=None)
    # parser.add_argument('-o', '--output', type=str,
    #                     help='Ouput file name root.',
    #                     default=None)
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

    # Check aux files folder
    if args.source is None:
        args.source = os.path.join('.', 'tmp')

    # # Check output filename
    # if args.output is None:
    #     args.output = 'DumpFile'+str(datetime.now())

    return args


def main():
    """Main labels_assignment.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: assign_labels.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* Aux files source folder: {}'.format(args.source))
    print('* Output files: {}.[pkl, csv, ...]'.format(args.output))
    print('-------------------------------------------------------------------')




################################################################################

if __name__ == '__main__':
    main()
