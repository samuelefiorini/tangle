"""This module has all the functions needed to detect diabetics."""
import os
import warnings

import numpy as np
import pandas as pd
from mbspbs10pc import __path__ as home
from tqdm import tqdm

warnings.filterwarnings('ignore')


def find_metafter(dd, pos_id, target_year=2012):
    """"Find the people on metformin + other drug.

    This function finds the patients that were prescribed to some non-metformin
    diabetes drug after an initial metformin prescription.

    Remark: There is no difference between someone that changes metformin for a
    new drug, and someone who end up using both the drugs.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    pos_id: pd.DataFrame
        The DataFrame generated from the output of find_positive_samples().

    target_year: integer (default=2012)
        The target year

    Returns:
    --------------
    out: dictionary
        Dictionary having target patient IDs (metformin + other drug) as keys
        and SPPLY_DT as values.
    """
    # Take only the relevant dd values
    _dd = dd['PBS_SAMPLE_10PCT_'+str(target_year)+'.csv']

    # Load the metformin items
    met_items = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin_items.csv'),
                    header=0).values.ravel())

    # Build the DataFrame of interest
    df = pd.DataFrame(columns=['PTNT_ID', 'ITM_CD', 'SPPLY_DT'])

    # Iterate on the positive indexes and fill up the data frame
    i = 0
    for idx in pos_id.index:
        for itm, dt in zip(_dd[idx]['ITM_CD'], _dd[idx]['SPPLY_DT']):
            df.loc[i, 'PTNT_ID'] = idx
            df.loc[i, 'ITM_CD'] = itm
            df.loc[i, 'SPPLY_DT'] = pd.to_datetime(dt, format='%d%b%Y')
            i += 1

    # Keep only patients that have at least one metformin prescription
    metonce = []
    for _, row in df.iterrows():
        if row['ITM_CD'] in met_items:
            metonce.append(row['PTNT_ID'])
    metonce = set(metonce)

    # Iterate on them
    metafter = []
    spply_dt = []
    for idx in metonce:
        tmp = df.loc[df['PTNT_ID'] == idx, ['ITM_CD', 'SPPLY_DT']]
        # Sort by date
        tmp.sort_values(by='SPPLY_DT', inplace=True)
        # Get where the metformin was prescribed
        mask = [s in met_items for s in tmp['ITM_CD']]
        mask = np.where(map(lambda x: not x, mask))[0]
        # If the non-metformin drug is prescribed after the position 0
        # it is likely that the patient started to take a new medication
        if len(mask) > 0 and not (mask[0] == 0):
            metafter.append(idx)
            spply_dt.append(tmp['SPPLY_DT'].values[mask[0]])
            # use the first non-metformin prescription as supply date

    # Retrieve the metafter subjects and create the output dictionary
    # out = {idx: min(df.loc[df['PTNT_ID'] == idx, 'SPPLY_DT']) for idx in metafter}
    out = {idx: dt for idx, dt in zip(metafter, spply_dt)}

    return out


def find_metonly(dd, pos_id, target_year=2012):
    """"Find the people on metformin only.

    Parameters:
    --------------
    dd: dictionary
        The output of find_diabetics().

    pos_id: pd.DataFrame
        The DataFrame generated from the output of find_positive_samples().

    target_year: integer (default=2012)
        The target year

    Returns:
    --------------
    out: dictionary
        Dictionary having target patient IDs (metformin only) as keys and
        SPPLY_DT as values.
    """
    # Load the metformin items
    met_items = set(pd.read_csv(os.path.join(home[0], 'data', 'metformin_items.csv'),
                    header=0).values.ravel())

    # Build the DataFrame of interest
    df = pd.DataFrame(columns=['PTNT_ID', 'ITM_CD', 'SPPLY_DT'])

    # Iterate on the positive indexes and fill up the data frame
    i = 0
    for idx in pos_id.index:
        for itm, dt in zip(_dd[idx]['ITM_CD'], _dd[idx]['SPPLY_DT']):
            df.loc[i, 'PTNT_ID'] = idx
            df.loc[i, 'ITM_CD'] = itm
            df.loc[i, 'SPPLY_DT'] = dt
            i += 1

    # Keep the pantients in metformin ONLY
    metonly = []
    for idx in df['PTNT_ID']:
        items = df.loc[df['PTNT_ID'] == idx, 'ITM_CD'].values.ravel().tolist()
        if set(items).issubset(met_items):
            metonly.append(idx)

    # Retrieve the metonly subjects and create the output dictionary
    out = {}
    for idx in metonly:
        # (but change to the correct datetime format first)
        out[idx] = min(map(lambda x: pd.to_datetime(x, format='%d%b%Y'),
                           df.loc[df['PTNT_ID'] == idx, 'SPPLY_DT']))

    return out


def find_diabetics(pbs_files, ccc=set()):
    """Search people using diabetes drugs in input PBS files.

    Parameters:
    --------------
    pbs_files: list
        List of input PBS filenames.

    ccc: set
        Set of continuoly and consistently concessional subjects.

    Returns:
    --------------
    df: pandas.DataFrame
        Table containing only the records of ccc using diabetics drugs.
    """
    # Load the drugs used in diabetes list file
    _dd = pd.read_csv(os.path.join(home[0], 'data', 'drugs_used_in_diabetes.csv'),
                      header=0)

    # Fix 6-digit notation
    dd = set()  # dd should be a set for performance reasons
    for item in _dd.values.ravel():
        if len(item) < 6:
            dd.add(str(0)+item)
        else:
            dd.add(item)

    # Find the ccc diabetics and save them in a single data frame
    columns = ['ITM_CD', 'PTNT_ID', 'SPPLY_DT']
    df = pd.DataFrame(columns=columns)
    for pbs in tqdm(pbs_files, desc='PBS files loading', leave=False):
        pbs_df = pd.read_csv(pbs, header=0, engine='c',
                             usecols=columns)
        pbs_df = pbs_df[pbs_df['PTNT_ID'].isin(ccc)]  # keep only ccc
        pbs_df = pbs_df[pbs_df['ITM_CD'].isin(dd)]  # keep only diabetics
        df = pd.concat((df, pbs_df))
    df.loc[:, 'SPPLY_DT'] = pd.to_datetime(df['SPPLY_DT'], format='%d%b%Y')

    return df
