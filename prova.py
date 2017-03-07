from recommenders.Mixed_Markov_Recommender import MixedMarkovChainRecommender
from recommenders.Freq_Seq_Mining_Recommender import FreqSeqMiningRecommender
import pandas as pd

from util.data_expansion import user_profile_expansion
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
p=10000
db=pd.read_csv('datasets/sequenceDb_'+str(p)+'.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()
seqs = list(filter(lambda x: len(x)>1,seqs))
seqs = list(map(lambda x: list(map(lambda y:str(y),x)),seqs))
model = gensim.models.Word2Vec(seqs, min_count=1,sg=1,workers=8)

model.most_similar(positive=['3367'],topn=3)


recFreq = FreqSeqMiningRecommender(0.002,0.1,10,1,spmf_path='spmf/spmf.jar',db_path='sequences.txt')
recFreq.fit([])
recFreq.get_freq_seqs()
recFreq.recommend(['748'])

from sklearn import tree
from recommenders.Supervised_Recommender import SupervisedRecommender
from util.split import balance_dataset
from util.data_expansion import data_expansion,user_profile_expansion
clf = tree.DecisionTreeClassifier()
r = SupervisedRecommender(1)
sequences = [[1,2,3,4],[9,7,4],[3,2,1],[0,4,3,2]]
s= sequences * 100
r.fit(s)
r.recommend([1,2,3])
r.recommend([2,3])
r.recommend([9,7])
r.recommend([3,2])
r.recommend([2])
r.recommend([2,3,1,4,7])

def _split_train_test(data,col_index,n_unique_items):
    test = data[:,col_index]
    train = data[:,[x for x in range(data.shape[1]) if x >= n_unique_items]]
    return train,test
data,mapping = data_expansion(s,1)
train,test = _split_train_test(data,0,len(mapping))
train,test = balance_dataset(train,test)
tree = clf.fit(train.todense(),test.toarray().ravel())
u=user_profile_expansion([1,2,3],1,mapping)
tree.predict(u.toarray())

tree = clf.fit([[1,2,3,4],[1,2,3,3]],[1,0])
tree.predict([[1,2,3,3]])

from tqdm import tqdm
for i in tqdm(range(10000)):
    for j in range(10000):
        a=2**(i)