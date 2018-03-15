#!/usr/bin/env python
"""Extract population of interest from MBSPBS 10% dataset.

This script finds continuously and consistently concessional subjects in the
PBS files (2008 - 2014) that were prescribed to diabetes control drugs.
In order to be considered continuously and consistently concessional, a subject
must:

    1) continuously use the concessional cards, i.e.: they use it for at least
       50% of the observation years,
    2) consistently satisfy the condition at point 1), i.e.: for at least 50%
       of the PBS benefit items each year.

This script will then produce a list of continuously and consistently
concessional subjects that use diabetes control drugs.
"""

from __future__ import print_function

import argparse
import os
from datetime import datetime

import joblib as jl
from mbspbs10pc import concessionals_utils as c_utils
from mbspbs10pc import diabete_utils as d_utils
from mbspbs10pc.utils import check_input


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
        args.output = 'DumpFile'+str(datetime.now())

    return args


def main():
    """Main get_population_of_interest.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: get_population_of_interest.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* Output files: {}.[pkl, csv, ...]'.format(args.output))
    print('-------------------------------------------------------------------')

    # PBS 10% dataset files
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    pbs_files_fullpath = [os.path.join(args.root, pbs) for pbs in pbs_files]

    # --- STEP 1 --- #
    # Find the continuously concessionals (Condition #1)
    filename = args.output+'_cont_.pkl'
    if not os.path.exists(filename):
        print('* Looking for continuously concessionals ...')
        cont_conc = c_utils.find_continuously_concessionals(pbs_files_fullpath)
        print('* Saving {} '.format(filename), end=' ')
        jl.dump(cont_conc, open(filename, 'wb'))
        print(u'\u2713')
    else:
        cont_conc = jl.load(open(filename, 'rb'))
    print('* {} Subjects continuously use concessional cards'.format(len(cont_conc)))

    # --- STEP 2 --- #
    # Filter out the subjects that are not using the concessional cards for at
    # least 50% of the times for each year (Condition #2)
    filename = args.output+'_cons_.pkl'
    if not os.path.exists(filename):
        print('* Looking for consistently concessionals ...')
        cons_conc = c_utils.find_consistently_concessionals(pbs_files_fullpath)
        print('* Saving {} '.format(filename), end=' ')
        jl.dump(cons_conc, open(filename, 'wb'))
        print(u'\u2713')
    else:
        cons_conc = jl.load(open(filename, 'rb'))
    print('* {} Subjects consistently use concessional cards'.format(len(cons_conc)))

    # --- STEP 3 --- #
    # Intersect the two sets and get the consistently and continuous
    # concessional cards users
    filename = args.output+'_cc_.pkl'
    if not os.path.exists(filename):
        ccc = cons_conc.intersection(cont_conc)
        print('* Saving {} '.format(filename), end=' ')
        jl.dump(ccc, open(filename, 'wb'))
        print(u'\u2713')
    else:
        ccc = jl.load(open(filename, 'rb'))
    print('* {} Subjects consistently AND continuously '
          'use concessional cards'.format(len(ccc)))

    # --- STEP 4 --- #
    # Find continuously and consistently concessional people on diabetic drugs
    filename = args.output+'_dd_.pkl'
    if not os.path.exists(filename):
        print('* Looking for subjects on diabete control drugs ...')
        dd, subjs = d_utils.find_diabetics(pbs_files_fullpath, ccc)
        print('\n* Saving {} '.format(filename), end=' ')
        jl.dump({'dd': dd, 'subjs': subjs}, open(filename, 'wb'))
        print(u'\u2713')
    else:
        tmp = jl.load(open(filename, 'rb'))
        dd, subjs = tmp['dd'], tmp['subjs']
    print('* {} Subjects consistently and continuously concessional '
          'use diabetes control drugs'.format(len(subjs)))


################################################################################

if __name__ == '__main__':
    main()
