from recommenders.FreqSeqMiningRecommender import FreqSeqMiningRecommender
from recommenders.PopularityRecommender import PopularityRecommender
from recommenders.MixedMarkovRecommender import MixedMarkovChainRecommender
from recommenders.Prod2VecRecommender import Prod2VecRecommender
from util.split import random_holdout,temporal_holdout
from util import evaluation,metrics
import logging
import argparse
from collections import OrderedDict
from util.createSeqDb import create_seq_db_filter_top_k,from_seqs_to_spmfdb



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', PopularityRecommender),
    ('FPM', FreqSeqMiningRecommender),
    ('Markov', MixedMarkovChainRecommender),
    ('Prod2Vec', Prod2VecRecommender)
])

available_holdout_methods = OrderedDict([
    ('random_holdout', random_holdout),
    ('temporal_holdout', temporal_holdout)
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,help='Dataset path, format: session_id,user_id(discarded),item_id,timestamp')
parser.add_argument('--only_top_k', type=int,default=10000,help='Number of unique items in the db to consider')
parser.add_argument('--holdout_method', type=str,default='random_holdout',help='random_holdout')
parser.add_argument('--train_perc', type=int,default=0.8,help='Percentage of dataset for training')
parser.add_argument('--recommender', type=str, default='top_pop')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--last_k', type=int, default=1)
args = parser.parse_args()

# get the recommender class
assert args.recommender in available_recommenders, 'Unknown recommender: {}'.format(args.recommender)
assert args.holdout_method in available_holdout_methods, 'Unknown holdout method: {}'.format(args.holdout_method)

RecommenderClass = available_recommenders[args.recommender]
holdout_method = available_holdout_methods[args.holdout_method]

# parse recommender parameters
init_args = OrderedDict()
if args.params:
    for p_str in args.params.split(','):
        key, value = p_str.split('=')
        try:
            init_args[key] = eval(value)
        except:
            init_args[key] = value

logging.info('Loading data')
seqs = list(create_seq_db_filter_top_k(args.dataset,args.only_top_k)['sequence'])
seqs = list(filter(lambda x: len(x) > args.last_k,seqs)) #filter too short
seqs = list(map(lambda x: list(map(lambda y:str(y),x)),seqs)) #as strings

# split dataset
logging.info("Splitting train and test:" + str(args.train_perc))
train_seq,test_seq = holdout_method(seqs, args.train_perc)
logging.info("Train size:{} test size:{}".format(len(train_seq),len(test_seq)))

# create db for FPM
if args.recommender =='FPM' and 'spmf_path' in  init_args:
    logging.info('Creating db for SPMF')
    db_fout = from_seqs_to_spmfdb(train_seq)
    init_args['db_path']=db_fout


# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Recommender: {}'.format(recommender))
recommender.fit(train_seq)

# evaluate the ranking quality
logger.info('Ranking quality')
p,r = evaluation.set_evaluation(recommender,test_seq,args.last_k,'total',[metrics.precision,metrics.recall])
logger.info('Set evaluation - Precision:{}, Recall:{}'.format(p,r))
p,r = evaluation.sequential_evaluation(recommender,test_seq,args.last_k,'total',[metrics.precision,metrics.recall])
logger.info('Sequential evaluation - Precision:{}, Recall:{}'.format(p,r))
