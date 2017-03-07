import random
from scipy.sparse import find

def random_holdout(seqs, perc=0.8,seed=1234):
    seqs = seqs.sample(frac=1,random_state=seed)
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
    raise NameError('Not implemented')
    train_split =[]
    test_split =[]
    # split data according to the shuffled index and the holdout size
    for i,x in enumerate(seqs):
        if timestamps[i] < ts_threshold:
            train_split.append(x)
        else:
            test_split.append(x)

    return train_split, test_split

def balance_dataset(x,y):

    number_of_elements = y.shape[0]
    nnz = set(find(y)[0])
    zero = set(range(number_of_elements)).difference(nnz)

    max_samples = min(len(zero),len(nnz))

    nnz_indices = random.sample(nnz,max_samples)
    zero_indeces = random.sample(zero,max_samples)
    indeces = nnz_indices + zero_indeces

    return x[indeces,:],y[indeces,:]
