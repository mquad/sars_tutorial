
class ISeqRecommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, X):
        pass


    def recommendSet(self, sequence):
        pass

    def recommendSequence(self,sequence):
        pass
