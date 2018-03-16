#!/usr/bin/env python
"""Assign class labels to the population of interest.

This script will then produce a list of continuously and consistently
concessional subjects that use metformin. The following labels are assigned:

    * MET_ONLY, i.e.: patients that are using metformin ONLY
    * MET+X, i.e.: patients that after a first metformin prescription
      started to use another diabetes controlling drug.
    * MET2X, i.e.: patients that changed from metformin to another drug
"""

from __future__ import print_function

import argparse
import os
from datetime import datetime

import joblib as jl
import numpy as np
import pandas as pd
from mbspbs10pc import diabete_utils as d_utils
from mbspbs10pc.utils import check_input
from mbspbs10pc.utils import flatten


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

    # Check aux files folder
    if args.source is None:
        args.source = os.path.join('.', 'tmp')

    # Check output filename
    if args.output is None:
        args.output = 'DumpFile'+str(datetime.now())

    return args


def main():
    """Main assign_labels.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: assign_labels.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* Aux files source folder: {}'.format(args.source))
    print('* Output files: {}'.format(args.output))
    print('-------------------------------------------------------------------')

    # Load the diabetic subjects info
    dd_file = filter(lambda x: '_dd_' in x, os.listdir(args.source))[0]
    print('* Loading {}...'.format(dd_file), end=' ')
    dd = jl.load(open(os.path.join(args.source, dd_file), 'rb'))
    print(u'\u2713')

    # Init the label file
    # the LABEL columns can be:
    # - METONLY for meformin only
    # - MET+X for people using both metformin and another drug
    # - MET2X or patients that changed metformin for another drug
    # START_DATE and END_DATE are the sequence exraction date, which are different
    # according to the label:
    #           [METONLY] from the first metformin prescription to the end
    #           [OTHER] from the first metformin to the first non metformin
    labels = pd.DataFrame(index=np.unique(dd['PTNT_ID']),
                          columns=['LABEL', 'START_DATE', 'END_DATE'])

    # Find patients on metformin ONLY
    print('* Looking for METONLY...')
    idx0, start_date0, end_date0 = d_utils.find_metonly(dd)
    labels.loc[idx0, 'START_DATE'] = start_date0
    labels.loc[idx0, 'END_DATE'] = end_date0
    labels.loc[idx0, 'LABEL'] = 'METONLY'
    print('* {} METONLY subjects'.format(len(idx0)), end=' ')
    print(u'\u2713')

    # Find patients using both metformin and other drugs
    print('* Looking for MET+X...')
    idx1, start_date1, end_date1 = d_utils.find_metx(dd, min_metformin=10)
    labels.loc[idx1, 'START_DATE'] = start_date1
    labels.loc[idx1, 'END_DATE'] = end_date1
    labels.loc[idx1, 'LABEL'] = 'MET+X'
    print('* {} MET+X subjects'.format(len(idx1)), end=' ')
    print(u'\u2713')

    # Find patients that changed from metformin to other drugs
    print('* Looking for MET2X...')
    idx2, start_date2, end_date2 = d_utils.find_met2x(dd, min_metformin=10)
    labels.loc[idx2, 'START_DATE'] = start_date2
    labels.loc[idx2, 'END_DATE'] = end_date2
    labels.loc[idx2, 'LABEL'] = 'MET2X'
    print('* {} MET2X subjects'.format(len(idx2)), end=' ')
    print(u'\u2713')

    # Fill the holes
    print('* Evaluating the other samples...')
    idx3, start_date3, end_date3 = d_utils.find_others(dd, met_idx=labels.dropna().index)
    labels.loc[idx3, 'START_DATE'] = start_date3
    labels.loc[idx3, 'END_DATE'] = end_date3
    labels.loc[idx3, 'LABEL'] = 'OTHER'
    print('* {} OTHER subjects'.format(len(idx3)), end=' ')
    print(u'\u2713')

    # Sanity checks - for peace of mind -
    assert(len(set(idx0).intersection(set(idx1))) == 0)
    assert(len(set(idx0).intersection(set(idx2))) == 0)
    assert(len(set(idx0).intersection(set(idx3))) == 0)
    assert(len(set(idx1).intersection(set(idx2))) == 0)
    assert(len(set(idx1).intersection(set(idx3))) == 0)
    assert(len(set(idx2).intersection(set(idx3))) == 0)

    # Save the labels
    ext = '.csv' if not args.output.endswith('.csv') else ''
    output_filename = args.output + ext
    print('* Saving {}...'.format(output_filename), end=' ')
    labels.to_csv(output_filename)
    print(u'\u2713')

################################################################################

if __name__ == '__main__':
    main()
