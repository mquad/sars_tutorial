import pandas as pd
import ast
from collections import defaultdict

##scan and record sid first and last occurrence
db = pd.read_csv('datasets/prova.txt',converters={'songs':ast.literal_eval})
db.drop('ts',axis=1,inplace=True)

sids =defaultdict(set)
firstAndLastOccurrence=defaultdict(lambda:defaultdict(lambda :([],[])))

#for each row fill in sids
for sid,row in db.iterrows():
    for pos,item in enumerate(row['songs']):
        sids[item].add(sid)
        if not firstAndLastOccurrence[item][sid][0]:
            firstAndLastOccurrence[item][sid][0].append(pos)
            firstAndLastOccurrence[item][sid][1].append(pos)
        firstAndLastOccurrence[item][sid][1][0] = pos