import logging
from recommenders.ISeqRecommender import ISeqRecommender
from util.markov.Markov import add_nodes_to_graph,add_edges,add_fractional_count,apply_clustering

"""Implementation from Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4"""

class MarkovChainRecommender(ISeqRecommender):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


    def __init__(self,last_k):
        super(MarkovChainRecommender, self).__init__()
        self.last_k = last_k

    def fit(self,seqs):
        """Takes a list of list of seqeunces ."""

        self.tree,self.count_dict,self.G = add_nodes_to_graph(self.last_k)
        self.G = add_edges(self.tree,self.count_dict,self.G,self.last_k)
        self.G = add_fractional_count(self.G,self.last_k,seqs)
        self.G,_,_ = apply_clustering(self.G)