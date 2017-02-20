import argparse
import logging
from collections import OrderedDict
from recommenders import FreqSeqMining
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('freq_seqs', FreqSeqMining)
])

##laod data
import ast
db=pd.read_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()

from pymining import seqmining
freq_seqs = seqmining.freq_seq_enum(seqs, 1000)

