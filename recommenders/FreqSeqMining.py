import recommenders
from pymining import seqmining
from tree.Tree import SmartTree

class FreqSeqMiningRecommender(recommenders.ISeqRecommender):
    """Frequent sequence mining recommender"""


    def __init__(self):
        super(FreqSeqMiningRecommender, self).__init__()

    def fit(self,seqs,minsup=1000):
        freq_seqs = seqmining.freq_seq_enum(seqs, minsup)

        self.rootNode = 0
        self.tree = SmartTree()
        for tuple in freq_seqs:
            if len(tuple[0]) == 1:
                #add node to root
                self.tree.create_node(tuple[0][0],parent=self.rootNode,data={"support":tuple[1]})
            elif len(tuple[0]) > 1:
                #add entire path starting from root
                self.tree.add_path((self.rootNode,) + tuple[0],tuple[1],self.rootNode)
            else:
                raise NameError('Frequent sequence of length 0')
