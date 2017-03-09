from recommenders.Freq_Seq_Mining_Recommender import FreqSeqMiningRecommender
from recommenders.Popularity_Recommender import PopularityRecommender
from recommenders.Mixed_Markov_Recommender import MixedMarkovChainRecommender
from recommenders.Prod2Vec_Recommender import Prod2VecRecommender
from recommenders.Supervised_Recommender import SupervisedRecommender
from recommenders.FPMC_Recommender import FPMCRecommender
from util.split import random_holdout,temporal_holdout
from util import evaluation,metrics
import logging
import argparse
from collections import OrderedDict
from util.createSeqDb import create_seq_db_filter_top_k,from_seqs_to_spmfdb
from functools import reduce

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', PopularityRecommender),
    ('FPM', FreqSeqMiningRecommender),
    ('Markov', MixedMarkovChainRecommender),
    ('Prod2Vec', Prod2VecRecommender),
    ('Supervised',SupervisedRecommender),
    ('FPMC',FPMCRecommender)
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
parser.add_argument('--top_n_list', type=str)
parser.add_argument('--last_months', type=int,default=12)
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

# parse top n list
top_n_lst = []
if args.top_n_list:
    for p_str in args.top_n_list.split(','):
        top_n_lst.append(int(p_str))

logging.info('Loading data')
data = create_seq_db_filter_top_k(args.dataset,args.only_top_k,args.last_months)
data = data[data['sequence'].map(len) > abs(args.last_k)]
data['sequence'] = data['sequence'].map((lambda x: list(map(lambda y:str(y),x))))

# split dataset
logging.info("Splitting train and test:" + str(args.train_perc))
train_data,test_data = holdout_method(data, args.train_perc)
logging.info("Train size:{} test size:{}".format(len(train_data),len(test_data)))
logging.info("Average sequence length:{}".format(reduce(lambda x,y:x+y,list(map(len,list(data['sequence']))))/len(list(data['sequence']))))

# create db for FPM
if args.recommender =='FPM' and 'spmf_path' in  init_args:
    logging.info('Creating db for SPMF')
    db_fout = from_seqs_to_spmfdb(list(train_data['sequence']))
    init_args['db_path']=db_fout


# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Fitting Recommender: {}'.format(recommender))
if args.recommender == 'FPMC':
    recommender.declare(data)
    recommender.fit(train_data)
else:
    recommender.fit(list(train_data['sequence']))


# evaluate the ranking quality
for n in top_n_lst:
    logger.info('Ranking quality top_n: '+str(n))
    if args.recommender == 'FPMC':
        p,r = evaluation.set_evaluation_use_user(recommender,list(test_data['sequence']),list(test_data['user_id']),args.last_k,'total',[metrics.precision,metrics.recall],n)
        logger.info('Set evaluation - Precision:{}, Recall:{}'.format(p,r))
        p,r = evaluation.sequential_evaluation_use_user(recommender,list(test_data['sequence']),list(test_data['user_id']),args.last_k,'total',[metrics.precision,metrics.recall],n)
        logger.info('Sequential evaluation - Precision:{}, Recall:{}'.format(p,r))
    else:
        p,r = evaluation.set_evaluation(recommender,list(test_data['sequence']),args.last_k,'total',[metrics.precision,metrics.recall],n)
        logger.info('Set evaluation - Precision:{}, Recall:{}'.format(p,r))
        p,r = evaluation.sequential_evaluation(recommender,list(test_data['sequence']),args.last_k,'total',[metrics.precision,metrics.recall],n)
        logger.info('Sequential evaluation - Precision:{}, Recall:{}'.format(p,r))
