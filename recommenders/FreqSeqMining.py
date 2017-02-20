from recommenders import ISeqRecommender
from pymining import seqmining,assocrules

class FreqSeqMiningRecommender(ISeqRecommender):
    """Frequent sequence mining recommender"""


####
list(map(lambda x:''.join(x),list(map(lambda x:x[0],freq_seqs))))