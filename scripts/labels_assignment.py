#!/usr/bin/env python
"""Find concessional subjects in the PBS files.

This script finds continuously and consistently concessional subjects in the
PBS files (2008 - 2014) that were prescribed to diabetes controlling drugs.
In order to be considered continuously and consistently concessional, a subject
must:

    1) continuously use the concessional cards, i.e.: they use it for at least
       50% of the observation years,
    2) consistently satisfy the condition at point 1), i.e.: for at least 50%
       of the PBS benefit items each year.

This script will then produce a list of continuously and consistently
concessional subjects that use diabetes controlling drugs. If the option
`-m` (or, equivalently `--metformin`) two more labels will be created:

    a) MET_ONLY, i.e.: patients that are using metformin ONLY
    b) MET_AFTER, i.e.: patients that after a first metformin prescription
       started to use another diabetes controlling drug.
"""

from __future__ import print_function

import argparse
import cPickle as pkl
import os
from datetime import datetime

import joblib as jl
import pandas as pd
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
    parser.add_argument('-t', '--target_year', type=int,
                        help='Diabetes drug starting year.', default=2012)
    parser.add_argument('-o', '--output', type=str,
                        help='Ouput file name root.',
                        default=None)
    parser.add_argument('-m', '--metformin', action='store_true',
                        help='Assign the metformin-related labels: i.e.: '
                        'MET_ONLY and MET_AFTER')
    parser.add_argument('-sic', '--skip_input_check', action='store_false',
                        help='Skip the input check (default=False).')
    parser.add_argument('-nj', '--n_jobs', type=int,
                        help='The number of processes to use.', default=4)
    # parser.add_argument('-cs', '--chunk_size', type=int,
    #                     help='The numer of rows each process has access to.',
    #                     default=1000)
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
    """Main labels_assignment.py routine."""
    print('-------------------------------------------------------------------')
    print('MBS - PBS 10% dataset utility: labels_assignment.py')
    print('-------------------------------------------------------------------')
    args = init_main()

    print('* Root data folder: {}'.format(args.root))
    print('* Target year: {}'.format(args.target_year))
    print('* Output files: {}.[pkl, csv, ...]'.format(args.output))
    print('* Number of jobs: {}'.format(args.n_jobs))
    print('[{}] Metformin labels: {}'.format(*('+', 'ON') if args.metformin else (' ', 'OFF')))
    # print('* Chunk size: {}'.format(args.chunk_size))
    print('-------------------------------------------------------------------')

    # PBS 10% dataset files
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    pbs_files_fullpath = [os.path.join(args.root, pbs) for pbs in pbs_files]

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

    # Filter out the subjects that are not using the concessional cards for at
    # least 50% of the times for each year
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

    # Intersect the two sets and get the consistently and continuous
    # concessional cards users
    filename = args.output+'_cc_.pkl'
    if not os.path.exists(filename):
        cons_cont_conc = cons_conc.intersection(cont_conc)
        print('* Saving {} '.format(filename), end=' ')
        jl.dump(cons_cont_conc, open(filename, 'wb'))
        print(u'\u2713')
    else:
        cons_cont_conc = jl.load(open(filename, 'rb'))
    print('* {} Subjects consistently AND continuously '
          'use concessional cards'.format(len(cons_cont_conc)))

    # Find people on diabetic drugs
    filename = args.output+'_dd_.pkl'
    if not os.path.exists(filename):
        print('* Looking for subjects on diabete control drugs ...')  # progress bar embedded
        dd = d_utils.find_diabetics(pbs_files_fullpath,
                                    filter_copayments=False,
                                    n_jobs=args.n_jobs)
        print('\n* Saving {} '.format(filename), end=' ')
        jl.dump(dd, open(filename, 'wb'))
        print(u'\u2713')
    else:
        dd = jl.load(open(filename, 'rb'))

    # Find, for each year, the number of people that are continuously and
    # consistently using their concessional cards and that STARTED taking
    # drugs for diabetes; i.e.: people that are prescribed to diabetes drugs in
    # the current year and that were never prescribed before
    # This is our POSITIVE class.
    filename = args.output+'_{}_class_1.csv'.format(args.target_year)
    if not os.path.exists(filename):
        print('* Looking for the positive samples ...')  # progress bar embedded
        pos_id = d_utils.find_positive_samples(dd, cons_cont_conc,
                                               target_year=args.target_year)
        print('* Saving {}'.format(filename), end=' ')
        pd.DataFrame.from_dict(pos_id, orient='index').rename({0: 'SPPLY_DT'}, axis=1).to_csv(filename)
        print(u'\u2713')
    else:
        pos_id = pd.read_csv(filename, header=0, index_col=0)
    print('* I found {} positive samples'.format(len(pos_id)))

    # Find, among these people, the ones that are on metformin ONLY
    if args.metformin:
        filename = args.output+'_{}_METONLY_class_1.csv'.format(args.target_year)
        if not os.path.exists(filename):
            metonly = d_utils.find_metonly(dd, pos_id, target_year=args.target_year)
            print('* Saving {}'.format(filename), end=' ')
            pd.DataFrame.from_dict(metonly, orient='index').rename({0: 'SPPLY_DT'}, axis=1).to_csv(filename)
            print(u'\u2713')
        else:
            metonly = pd.read_csv(filename, header=0)
        print('* I found {} samples on metformin only'.format(len(metonly)))

    # DEBUG
    import sys
    sys.exit(-1)
    # DEBUG

    # Find, among these people, the ones that started on metformin and then
    # another drug was added
    # TODO

    # Find people that are continuously and consistently concessional users but
    # were NEVER prescribed with diabetes control drugs in the years
    # (2008-2014).
    # This is our NEGATIVE class.
    filename = args.output+'_class_0.csv'
    if not os.path.exists(filename):
        print('* Looking for the negative samples ...')  # progress bar embedded
        neg_id = d_utils.find_negative_samples(pbs_files_fullpath, dd, cons_cont_conc)
        print('* Saving {}'.format(filename), end=' ')
        pd.DataFrame(data=neg_id, columns=['PTNT_ID']).to_csv(filename, index=False)
        print(u'\u2713')
    else:
        neg_id = pd.read_csv(filename, header=0, index_col=0)
    print('* I found {} negative samples'.format(len(neg_id)))

    # Sanity check: no samples should be in common between positive and negative class
    assert(len(set(pos_id).intersection(set(neg_id))) == 0)
    print('* Negative and positive class do not overlap', end=' ')
    print(u'\u2713')


################################################################################

if __name__ == '__main__':
    main()
