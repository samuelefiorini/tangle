#!/usr/bin/env python
"""Script to generate the supervised dataset D = (X, y).

Our goal here is to generate a the dataset D = (X, y) from the raw MBS-PBS 10%
dataset. We will use the PBS dataset to identify which subjects after some point
in time (in between 2008-2014) were prescribed to the use of some
glycaemia-control drugs (see ../data/drugs_used_in_diabetes.csv).
These individuals will be labeled as our positive class (y = 1). Subjects that,
as far as we can see, were never prescribed with diabetes controlling drug are
our negative class (y = 0).

Remarks:
--------------
* For each year we filter the PTNT_ID that were prescribed of a drug listed in
  `data/drugs_used_in_diabetes.csv`

* In April 2012 they started to record every PBS item, even the ones below the
  co-payment threshold. For consistency, it is possible to exclude from the
  counts the PBS items having total cost < co-payment(year).
  Where total cost is 'BNFT_AMT'+'PTNT_CNTRBTN_AMT'.
  Be aware that the threshold varies in the years.
  See data/co-payments_08-18.csv. This only holds for General Beneficiaries.

 * Monthly breakdown can be turned on for visualization purposes (see for
   instance the summary notebook). Anyway, once you have selected a target year
   a monthly breakdown is always run to find the sequences.
"""

from __future__ import print_function
import argparse
import cPickle as pkl
import datetime
import os

import mbspbs10pc.make_xy_utils as utils
import pandas as pd
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
                        help='Temporary file pickle output.',
                        default=None)
    parser.add_argument('-s', '--skip_input_check', action='store_false',
                        help='Skip the input check (default=False).')
    parser.add_argument('-fc', '--filter_copayments', action='store_true',
                        help='Use this option to include the PBS '
                        'items having total cost <= co-payment(year)'
                        ' (default=True).')
    parser.add_argument('-mb', '--monthly_breakdown', action='store_true',
                        help='Split records in different months (default=True).')
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
    print('* Target year: {}'.format(args.target_year))
    print('* Number of jobs: {}'.format(args.n_jobs))
    print('* Chunk size: {}\n'.format(args.chunk_size))
    print('[{}] Co-payment filter: {}'.format(*('+', 'ON') if args.filter_copayments else (' ', 'OFF')))
    print('[{}] Monthly breakdown: {}'.format(*('+', 'ON') if args.monthly_breakdown else (' ', 'OFF')))
    print('------------------------------------------')

    # MBS-PBS 10% dataset files
    mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    # sample_pin_lookout = filter(lambda x: x.startswith('SAMPLE'), os.listdir(args.root))[0]

    # Filter the population of people using drugs for diabetes
    pbs_files_fullpath = [os.path.join(args.root, '{}'.format(pbs)) for pbs in pbs_files]

    # Check output filename
    if args.output is None:
        filename = 'dumpFile'+str(datetime.now())+'.pkl'
    else:
        filename = args.output if args.output.endswith('.pkl') else args.output+'.pkl'

    # If the output file doesn't exist, scan the MBS-PBS dataset and create it
    if not os.path.exists(filename):
        print('* Looking for the positive samples ...')  # progress bar embedded
        dd = utils.find_population_of_interest(pbs_files_fullpath,
                                               filter_copayments=args.filter_copayments,
                                               monthly_breakdown=args.monthly_breakdown,
                                               chunksize=args.chunk_size,
                                               n_jobs=args.n_jobs)

        # Dump results
        print('* Saving {} '.format(filename), end=' ')
        pkl.dump(dd, open(filename, 'wb'))
        # print('done.\n')
        print(u'\u2713')
    else:
        # Otherwise just load it
        print('* Loading {} '.format(filename), end=' ')
        dd = pkl.load(open(filename, 'rb'))
        print(u'\u2713')

    # Find, for each year, the number of people that STARTED taking
    # drugs for diabetes; i.e.: people that are prescribed to diabetes drugs in
    # the current year and that were never prescribed before
    # This is our POSITIVE class.
    filename_y_1 = filename[:-4]+'_class_1.csv'
    if not os.path.exists(filename_y_1):
        pos_id = utils.find_positive_samples(dd, target_year=args.target_year)
        print('* Saving {}'.format(filename_y_1), end=' ')
        pd.DataFrame(data=pos_id, columns=['PTNT_ID']).to_csv(filename_y_1, index=False)
        print(u'\u2713')
    else:
        pos_id = pd.read_csv(filename_y_1, header=0).values.ravel()
    print('* I found {} positive samples'.format(len(pos_id)))

    # Find people that were NEVER prescribed with diabetes control drugs
    # in the years (2008-2014).
    # This is our NEGATIVE class.
    filename_y_0 = filename[:-4]+'_class_0.csv'
    if not os.path.exists(filename_y_0):
        print('* Looking for the negative samples ...')  # progress bar embedded
        neg_id = utils.find_negative_samples(pbs_files_fullpath, dd)
        print('* Saving {}'.format(filename_y_0), end=' ')
        pd.DataFrame(data=neg_id, columns=['PTNT_ID']).to_csv(filename_y_0, index=False)
        print(u'\u2713')
    else:
        neg_id = pd.read_csv(filename_y_0, header=0).values.ravel()
    print('* I found {} negative samples'.format(len(neg_id)))

    # Sanity check: no samples should be in common between positive and negative class
    assert(len(set(pos_id).intersection(set(neg_id))) == 0)
    print('* Negative and positive class do not overlap', end=' ')
    print(u'\u2713')

    # Now build the raw sequence data for each subject
    filename_x_1 = filename[:-4]+'_seq_0.csv'
    if not os.path.exists(filename_x_1):
        print('* Sequences extracion ...')
        pos_seq = utils.extract_sequences(mbs_files, pos_id)
        print('* Saving {}'.format(filename_x_1), end=' ')
        pd.DataFrame(data=pos_seq, columns=['Sequence'], index=pos_id).to_csv(filename_x_1)
        print(u'\u2713')
    else:
        pos_seq = pd.read_csv(filename_x_1, header=0, index_col=0)

    # -- then do the same for the negative class -- #



################################################################################

if __name__ == '__main__':
    main()
