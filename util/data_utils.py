import calendar
import datetime
import os
import time
from collections import Counter

import numpy as np
import pandas as pd


def create_seq_db_filter_top_k(path, topk=0, last_months=0):
    file = load_and_adapt(path, last_months=last_months)

    c = Counter(list(file['item_id']))

    if topk > 1:
        keeper = set([x[0] for x in c.most_common(topk)])
        file = file[file['item_id'].isin(keeper)]

    # group by session id and concat song_id
    groups = file.groupby('session_id')

    # convert item ids to string, then aggregate them to lists
    aggregated = groups['item_id'].agg({'sequence': lambda x: list(map(str, x))})
    init_ts = groups['ts'].min()
    users = groups['user_id'].min()  # it's just fast, min doesn't actually make sense

    result = aggregated.join(init_ts).join(users)
    result.reset_index(inplace=True)
    return result


def dataset_to_gru4rec_format(dataset):
    """
    Convert a list of sequences to GRU4Rec format.
    Based on this StackOverflow answer: https://stackoverflow.com/a/48532692

    :param dataset: the dataset to be transformed
    """

    lst_col = 'sequence'
    df = dataset.reset_index()
    unstacked = pd.DataFrame({
        col: np.repeat(df[col].values, df[lst_col].str.len()) for col in df.columns.drop(lst_col)}
    ).assign(**{lst_col: np.concatenate(df[lst_col].values)})[df.columns]
    # ensure that events in the session have increasing timestamps
    unstacked['ts'] = unstacked['ts'] + unstacked.groupby('user_id').cumcount()
    unstacked.rename(columns={'sequence': 'item_id'}, inplace=True)
    return unstacked


def sequences_to_spfm_format(sequences, tmp_path='tmp/sequences.txt'):
    """
    Convert a list of sequences to SPFM format and write them to `tmp_path`
    :param sequences: the list of sequences
    :param tmp_path: the path where sequences will be written in the SPFM format
    """
    basedir = os.path.split(tmp_path)[0]
    os.makedirs(basedir, exist_ok=True)
    with open(tmp_path, 'w') as fout:
        for s in sequences:
            fout.write(' -1 '.join(map(str, s)))
            fout.write(' -2\n')


def load_and_adapt(path, last_months=0):
    file_ext = os.path.splitext(path)[-1]
    if file_ext == '.csv':
        data = pd.read_csv(path, header=0)
    elif file_ext == '.hdf':
        data = pd.read_hdf(path)
    else:
        raise ValueError('Unsupported file {} having extension {}'.format(path, file_ext))

    col_names = ['session_id', 'user_id', 'item_id', 'ts'] + data.columns.values.tolist()[4:]
    data.columns = col_names

    if last_months > 0:
        def add_months(sourcedate, months):
            month = sourcedate.month - 1 + months
            year = int(sourcedate.year + month / 12)
            month = month % 12 + 1
            day = min(sourcedate.day, calendar.monthrange(year, month)[1])
            return datetime.date(year, month, day)

        lastdate = datetime.datetime.fromtimestamp(data.ts.max())
        firstdate = add_months(lastdate, -last_months)
        initial_unix = time.mktime(firstdate.timetuple())

        # filter out older interactions
        data = data[data['ts'] >= initial_unix]

    return data
