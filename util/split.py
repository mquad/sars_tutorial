import numpy as np
import random

def random_holdout(seqs, perc=0.8, seed=1234):
    # set the random seed
    random.seed(seed)
    #  shuffle data
    random.shuffle(seqs)
    nseqs = len(seqs)
    train_size = int(nseqs * perc)
    # split data according to the shuffled index and the holdout size
    train_split = seqs[:train_size]
    test_split = seqs[train_size:]

    return train_split, test_split

def temporal_holdout(seqs,timestamps, ts_threshold):
    """
    :param seqs:  list of sequences
    :param timestamps: ordered list of initial timestamps with respect to sequences
    :param ts_threshold: timestamp below wich a sequence is put in the train
    :return: train and test split
    """
    train_split =[]
    test_split =[]
    # split data according to the shuffled index and the holdout size
    for i,x in enumerate(seqs):
        if timestamps[i] < ts_threshold:
            train_split.append(x)
        else:
            test_split.append(x)

    return train_split, test_split

