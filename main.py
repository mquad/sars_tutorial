from recommenders.FreqSeqMining import FreqSeqMiningRecommender
import pandas as pd

##laod data
import ast
db=pd.read_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()

rec = FreqSeqMiningRecommender(0.0022,0.1,True)

rec.fit(seqs)
