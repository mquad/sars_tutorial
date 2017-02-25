from recommenders.FreqSeqMining import FreqSeqMiningRecommender
import pandas as pd
from util.split import random_holdout
from util import evaluation,metrics

##laod data
import ast
db=pd.read_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()

train_seq,test_seq = random_holdout(seqs, 0.8)

rec = FreqSeqMiningRecommender(0.005,0.1,10,1)
rec.fit(train_seq)

evaluation.set_evaluation(rec, test_seq, 1, 'total', [metrics.precision,metrics.recall], 10,1)


db="datasets/sequences.txt"
rec1 = FreqSeqMiningRecommender(0.003,0.1,10,2,"spmf/spmf.jar",db)
rec1.fit([])
len(rec1.get_freq_seqs())
rec = FreqSeqMiningRecommender(0.003,0.1,10,2)
a=list(map(lambda x:list(map(str,x)),seqs))
rec.fit(a)
