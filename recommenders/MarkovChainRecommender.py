import logging
from recommenders.ISeqRecommender import ISeqRecommender
from util.markov.Markov import add_nodes_to_graph,add_edges,add_fractional_count,apply_clustering

"""Implementation from Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4"""

class MarkovChainRecommender(ISeqRecommender):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def __init__(self,last_k):
        super(MarkovChainRecommender, self).__init__()
        self.last_k = last_k

    def fit(self,seqs):
        """Takes a list of list of seqeunces ."""

        logging.debug('Building Markov Chain model with k = '+str(self.last_k))
        self.tree,self.count_dict,self.G = add_nodes_to_graph(seqs,self.last_k)
        self.G = add_edges(self.tree,self.count_dict,self.G,self.last_k)
        self.G = add_fractional_count(self.G,self.last_k,seqs)
        self.G,_,_ = apply_clustering(self.G)

    def recommend(self, user_profile):

        #if the user profile is longer than the markov order, chop it keeping recent history
        state = tuple(user_profile[-self.last_k:])
        #see if graph has that state
        recommendations = []
        if self.G.has_node(state):
            #search for recommendations in the forward star
            rec_dict = {}
            for u,v in self.G.out_edges_iter([state]):
                lastElement =tuple(v[-1:])
                if lastElement in rec_dict:
                    rec_dict[lastElement] += self.G[u][v]['count']
                else:
                    rec_dict[lastElement] = self.G[u][v]['count']
            for k,v in rec_dict.items():
                recommendations.append((list(k),v))

        return recommendations

    def _set_graph_debug(self,G):
        self.G = G
