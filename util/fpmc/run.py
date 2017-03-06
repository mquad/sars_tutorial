import argparse
from random import shuffle

from utils import *

try:
    from FPMC_numba import FPMC
    print ('Using numba')
except ImportError:
    from util.fpmc import FPMC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='The directory of input', type=str)
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=15)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=32)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    args = parser.parse_args()

    f_dir = args.input_dir

    data_list, user_set, item_set = load_data_from_dir(f_dir)
    shuffle(data_list)

    train_ratio = 0.8
    split_idx = int(len(data_list) * train_ratio)
    tr_data = data_list[:split_idx]
    te_data = data_list[split_idx:]

    fpmc = FPMC(n_user=max(user_set) + 1, n_item=max(item_set) + 1,
                n_factor=args.n_factor, learn_rate=args.learn_rate, regular=args.regular)
    fpmc.user_set = user_set
    fpmc.item_set = item_set
    fpmc.init_model()

    acc, mrr = fpmc.learnSBPR_FPMC(tr_data, te_data, n_epoch=args.n_epoch, 
                                   neg_batch_size=args.n_neg, eval_per_epoch=False)

    print ("Accuracy:%.2f MRR:%.2f" % (acc, mrr))






