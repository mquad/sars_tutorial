import argparse
import logging
from collections import OrderedDict
from functools import reduce

from recommenders.FPMCRecommender import FPMCRecommender
from recommenders.FSMRecommender import FSMRecommender
from recommenders.MixedMarkovRecommender import MixedMarkovChainRecommender
from recommenders.PopularityRecommender import PopularityRecommender
from recommenders.Prod2VecRecommender import Prod2VecRecommender
from recommenders.SupervisedRecommender import SupervisedRecommender
from recommenders.RNNRecommender import RNNRecommender
from util import evaluation, metrics
from util.data_utils import create_seq_db_filter_top_k, sequences_to_spfm_format
from util.split import random_holdout, temporal_holdout

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', PopularityRecommender),
    ('FPM', FSMRecommender),
    ('Markov', MixedMarkovChainRecommender),
    ('Prod2Vec', Prod2VecRecommender),
    ('Supervised', SupervisedRecommender),
    ('FPMC', FPMCRecommender),
    ('RNN', RNNRecommender)
])

available_holdout_methods = OrderedDict([
    ('random_holdout', random_holdout),
    ('temporal_holdout', temporal_holdout)
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset path, columns: session_id, user_id, item_id, timestamp')
parser.add_argument('--only_top_k', type=int, default=10000, help='Number of unique items in the db to consider')
parser.add_argument('--holdout_method', type=str, default='random_holdout', help='Either random_holdout or temporal_holdout')
parser.add_argument('--train_perc', type=int, default=0.8, help='Percentage of dataset for training (for random_holdout only)')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--split_ts', type=int, help='Splitting timestamp (for temporal_holdout only)')
parser.add_argument('--recommender', type=str, default='top_pop')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--given_k', type=int, default=1)
parser.add_argument('--look_ahead', type=str, default='1')
parser.add_argument('--top_n_list', type=str)
parser.add_argument('--last_months', type=int, default=12)
args = parser.parse_args()

# get the recommender class
assert args.recommender in available_recommenders, 'Unknown recommender: {}'.format(args.recommender)
assert args.holdout_method in available_holdout_methods, 'Unknown holdout method: {}'.format(args.holdout_method)

RecommenderClass = available_recommenders[args.recommender]

# parse recommender parameters
init_args = OrderedDict()
if args.params:
    for p_str in args.params.split(','):
        key, value = p_str.split('=')
        try:
            init_args[key] = eval(value)
        except:
            init_args[key] = value

# parse top n list
top_n_lst = []
if args.top_n_list:
    for p_str in args.top_n_list.split(','):
        top_n_lst.append(int(p_str))

logging.info('Loading data')
data = create_seq_db_filter_top_k(args.dataset, args.only_top_k, args.last_months)

# split dataset
if args.holdout_method == 'random_holdout':
    logging.info("Randomly splitting sessions into train and test (perc={})".format(args.train_perc))
    train_data, test_data = random_holdout(data, args.train_perc, args.seed)
else:
    logging.info("Splitting session into train and test (split ts={})".format(args.split_ts))
    train_data, test_data = temporal_holdout(data, args.split_ts)

logging.info("Train size: {} - Test size: {}".format(len(train_data), len(test_data)))
logging.info("Average sequence length: {}".format(
    reduce(lambda x, y: x + y, list(map(len, list(data['sequence'])))) / len(list(data['sequence']))))

# remove sequences shorter than given_k from test data
test_data = test_data.loc[test_data['sequence'].map(len) > abs(args.given_k)]

# create db for FPM
if args.recommender == 'FPM' and 'spmf_path' in init_args:
    logging.info('Creating db for SPMF')
    sequences_to_spfm_format(list(train_data['sequence']), tmp_path='tmp/sequences.txt')
    init_args['db_path'] = 'tmp/sequences.txt'
    train_data = None   # loads training data from db_path

# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Fitting Recommender: {}'.format(recommender))
recommender.fit(train_data)

# evaluate the ranking quality
look_ahead = args.look_ahead
if look_ahead != 'all':
    look_ahead = int(look_ahead)
for n in top_n_lst:
    logger.info('Ranking quality top_n: ' + str(n))
    p, r = evaluation.sequential_evaluation(recommender,
                                            test_sequences=list(test_data['sequence']),
                                            users=list(test_data['user_id']),
                                            given_k=args.given_k,
                                            look_ahead=look_ahead,
                                            step=1,
                                            evaluation_functions=[metrics.precision, metrics.recall],
                                            scroll=True,
                                            top_n=n)
    logger.info('Sequential evaluation - Precision:{}, Recall:{}'.format(p, r))
    p, r = evaluation.sequential_evaluation(recommender,
                                            test_sequences=list(test_data['sequence']),
                                            users=list(test_data['user_id']),
                                            given_k=args.given_k,
                                            look_ahead=look_ahead,
                                            step=1,
                                            evaluation_functions=[metrics.precision, metrics.recall],
                                            scroll=False,
                                            top_n=n)
    logger.info('Sequential evaluation - Precision:{}, Recall:{}'.format(p, r))
