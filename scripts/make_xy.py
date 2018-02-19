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
"""

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
                        help='Temporary file pikle output.',
                        default=None)
    parser.add_argument('-s', '--skip_input_check', action='store_false',
                        help='Skip the input check (default=False).')
    parser.add_argument('-fc', '--filter_copayments', action='store_true',
                        help='Use this option to include the PBS '
                        'items having total cost <= co-payment(year)'
                        ' (default=True).')
    parser.add_argument('-mb', '--monthly_breakdown', action='store_true',
                        help='Split records in different months (default=True).')
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
    print('* Target year: {}\n'.format(args.target_year))
    print('[{}] Co-payment filter: {}'.format(*('+', 'ON') if args.filter_copayments else (' ', 'OFF')))
    print('[{}] Monthly breakdown: {}'.format(*('+', 'ON') if args.monthly_breakdown else (' ', 'OFF')))
    print('------------------------------------------')

    # MBS-PBS 10% dataset files
    # mbs_files = filter(lambda x: x.startswith('MBS'), os.listdir(args.root))
    pbs_files = filter(lambda x: x.startswith('PBS'), os.listdir(args.root))
    # sample_pin_lookout = filter(lambda x: x.startswith('SAMPLE'), os.listdir(args.root))[0]

    # Filter the population of people using drugs for diabetes
    pbs_files_fullpath = [os.path.join(args.root, '{}'.format(pbs)) for pbs in pbs_files]

    # dd = utils.find_population_of_interest(pbs_files_fullpath,
    #                                        filter_copayments=args.filter_copayments,
    #                                        monthly_breakdown=args.monthly_breakdown,
    #                                        chunksize=10000, n_jobs=32)
    #

    # Dump results
    if args.output is None:
        filename = 'DumpFile'+str(datetime.now())+'.pkl'
    else:
        filename = args.output if args.output.endswith('.pkl') else args.output+'.pkl'

    # print('- Saving {} ...'.format(filename))
    # with open(filename, 'wb') as f:  # FIXME
    #     pkl.dump(dd, f)
    # print('done.')

    print('- Loading {} ...'.format(filename))
    with open(filename, 'rb') as f:  # FIXME
        dd = pkl.load(f)
    print('done.')

    # Find, for each year, the number of people that STARTED taking
    # drugs for diabetes; i.e.: people that are prescribed to diabetes drugs in
    # the current year and that were never prescribed before
    pos_id = utils.find_positive_samples(dd, target_year=args.target_year)
    print('- {} positive samples'.format(len(pos_id)))

    # FIXME
    filename_1 = filename[:-4]+'_class_1.csv'
    print('- Saving {} ...'.format(filename_1))
    pd.DataFrame(data=pos_id, columns=['PTNT_ID']).to_csv(filename_1, index=False)
    print('done.')

    # Find people that were NEVER prescribed with diabetes control drugs
    neg_id = utils.find_negative_samples(pbs_files_fullpath, dd)
    print('- {} negative samples'.format(len(neg_id)))

    filename_0 = filename[:-4]+'_class_0.csv'
    print('- Saving {} ...'.format(filename_0))
    pd.DataFrame(data=neg_id, columns=['PTNT_ID']).to_csv(filename_0, index=False)
    print('done.')

    # Sanity check: no samples should be in common between positive and negative class
    assert(len(set(pos_id).intersection(set(neg_id))) == 0)




################################################################################

if __name__ == '__main__':
    main()
