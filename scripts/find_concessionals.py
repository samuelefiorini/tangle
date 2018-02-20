#!/usr/bin/env python
"""Find concessional subjects in the PBS files.

This script finds continuously and consistently concessional subjects in the
PBS files (2008 - 2014). In order to be considered continuously and
consistently concessional, a subject must:

    1) continuously use the concessional cards, i.e.: they use it for at least
       50% of the observation years,
    2) consistently satisfy the condition at point 1), i.e.: for at least 50%
       of the PBS benefit items each year.
"""

from __future__ import print_function

import argparse
import cPickle as pkl
import os
from datetime import datetime

import mbspbs10pc.find_concessionals_utils as utils
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
    parser.add_argument('-s', '--skip_input_check', action='store_false',
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
    """Main find_concessionas.py routine."""
    args = init_main()
    print('-----------------------------------------------------')
    print('MBS - PBS 10% dataset utility: find_concessionals.py')
    print('-----------------------------------------------------')

    print('* Root data folder: {}'.format(args.root))
    print('* Output files: {}.[pkl, csv, ...]'.format(args.output))

    # PBS 10% dataset files
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    pbs_files_fullpath = [os.path.join(args.root, pbs) for pbs in pbs_files]

    # Find the continuously concessionals (Condition #1)
    filename = args.output+'_cc_.pkl'
    if not os.path.exists(filename):
        print('* Looking for continuously concessionals ...')
        cont_conc = utils.find_continuously_concessionals(pbs_files_fullpath)
        print('* {} Subjects continuously use concessional cards'.format(len(cont_conc)))
        print('* Saving {} '.format(filename), end=' ')
        pkl.dump(cont_conc, open(filename, 'wb'))
        print(u'\u2713')
    else:
        cont_conc = pkl.load(open(filename, 'rb'))

    # Filter out the subjects that are not using the concessional cards for at
    # least 50% of the times for each year
    filename = args.output+'_filter_cc_.pkl'
    if not os.path.exists(filename):
        print('* Looking for consistently concessionals ...')
        cons_conc = utils.find_consistently_concessionals(pbs_files_fullpath, cont_conc)
        print('* {} Subjects consistently use concessional cards'.format(len(cons_conc)))
        print('* Saving {} '.format(filename), end=' ')
        pkl.dump(cons_conc, open(filename, 'wb'))
        print(u'\u2713')
    else:
        cons_conc = pkl.load(open(filename, 'rb'))








################################################################################

if __name__ == '__main__':
    main()
