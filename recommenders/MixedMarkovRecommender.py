import logging
from recommenders.ISeqRecommender import ISeqRecommender
from recommenders.MarkovChainRecommender import MarkovChainRecommender
"""Implementation from Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4"""

class MixedMarkovChainRecommender(ISeqRecommender):
    """
    Creates markov models with different values of k, and return recommendation by weighting the list of
    recommendation of each model
    """

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    recommenders = {}

    def __init__(self,from_k,to_k):
        super(MixedMarkovChainRecommender, self).__init__()
        self.from_k = from_k
        self.to_k = to_k
        #define the models
        for i in range(self.from_k,self.to_k+1):
            self.recommenders[i] = MarkovChainRecommender(i)

    def fit(self, X):
        for r in self.recommenders:
            r.fit(X)
