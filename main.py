import argparse
import logging
from collections import OrderedDict
from recommenders import FreqSeqMining
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('freq_seqs', FreqSeqMining)
])
