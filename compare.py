from recommenders.FreqSeqMiningRecommender import FreqSeqMiningRecommender
from recommenders.PopularityRecommender import PopularityRecommender
from recommenders.MixedMarkovRecommender import MixedMarkovChainRecommender
import pandas as pd
from util.split import random_holdout
from util import evaluation,metrics
import logging
import ast
from functools import reduce


def eval_rec(rec,test_seq):
        ev = evaluation.set_evaluation(rec, test_seq, last_k, 'total', [metrics.precision,metrics.recall])
        logging.info("set:"+str(ev))
        ev = evaluation.sequential_evaluation(rec, test_seq, last_k, 'total', [metrics.precision,metrics.recall])
        logging.info("sequence:"+str(ev))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

database_popular = [10,200,1000,5000,10000]
ks = [1,3,5,10]

for p in database_popular:
    logging.info('Loading data')
    db=pd.read_csv('datasets/sequenceDb_'+str(p)+'.csv',converters={'songs':ast.literal_eval})
    seqs = db['songs'].tolist()
    seqs = list(filter(lambda x: len(x)>1,seqs))

    logging.info("Average sequence length:{}".format(reduce(lambda x,y:x+y,list(map(len,seqs)))/len(seqs)))

    perc = 0.8
    logging.info("Splitting train and test:" + str(perc))
    train_seq,test_seq = random_holdout(seqs, perc)
    logging.info("Train size:{} test size:{}".format(len(train_seq),len(test_seq)))

    logging.info('Fit FPM')
    recFreq = FreqSeqMiningRecommender(0.002,0.1,10,1)
    recFreq.fit(train_seq)
    logging.info('Fit Markov')
    markovRec = MixedMarkovChainRecommender(1,5)
    markovRec.fit(train_seq)

    for last_k in ks:
        logging.info("Database_kept:{}, last_k:{}".format(p,last_k))

        popRec = PopularityRecommender(last_k)
        popRec.fit(train_seq)

        logging.info('Eval pop')
        eval_rec(popRec,test_seq)
        logging.info('------------------')
        logging.info('Eval FPM')
        eval_rec(recFreq,test_seq)
        logging.info('------------------')
        logging.info('Eval Markov')
        eval_rec(markovRec,test_seq)
        logging.info('------------------')


