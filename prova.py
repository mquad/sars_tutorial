from recommenders.MixedMarkovRecommender import MixedMarkovChainRecommender
import pandas as pd
from util.split import random_holdout
from util import evaluation,metrics
import logging
import ast
from functools import reduce


def eval_rec(rec,train_seq,test_seq):
        rec.fit(train_seq)
        ev = evaluation.set_evaluation(rec, test_seq, last_k, 'total', [metrics.precision,metrics.recall])
        logging.info(ev)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

p = 200
last_k = 1

logging.info('Loading data')
db=pd.read_csv('datasets/sequenceDb_'+str(p)+'.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()
seqs = list(filter(lambda x: len(x)>1,seqs))

logging.info("Average sequence length:{}".format(reduce(lambda x,y:x+y,list(map(len,seqs)))/len(seqs)))

perc = 0.8
logging.info("Splitting train and test:" + str(perc))
train_seq,test_seq = random_holdout(seqs, perc)
logging.info("Train size:{} test size:{}".format(len(train_seq),len(test_seq)))

logging.info("Database_kept:{}, last_k:{}".format(p,last_k))

markovRec = MixedMarkovChainRecommender(1,5)
markovRec.activate_debug_print()
logging.info('Eval Markov')
eval_rec(markovRec,train_seq,test_seq)
logging.info('------------------')

sentences=[['1','2','3'],['4','5','6']]
import gensim

# train word2vec on the two sentences
db=pd.read_csv('datasets/sequenceDb_'+str(p)+'.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()
seqs = list(filter(lambda x: len(x)>1,seqs))
model = gensim.models.Word2Vec(seqs, min_count=1,sg=1)


