import gc
import logging

from sars_tutorial.recommenders.ISeqRecommender import ISeqRecommender
from sars_tutorial.util.markov.Markov import add_nodes_to_graph, add_edges, apply_skipping, apply_clustering


class MarkovChainRecommender(ISeqRecommender):
    """
    Implementation from Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
    Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, order):
        """
        :param order: the order of the Markov Chain
        """
        super(MarkovChainRecommender, self).__init__()
        self.order = order

    def fit(self, train_data):
        sequences = train_data['sequence'].values

        logging.info('Building Markov Chain model with k = ' + str(self.order))
        logging.info('Adding nodes')
        self.tree, self.count_dict, self.G = add_nodes_to_graph(sequences, self.order)
        logging.info('Adding edges')
        self.G = add_edges(self.tree, self.count_dict, self.G, self.order)
        logging.info('Applying skipping')
        self.G = apply_skipping(self.G, self.order, sequences)
        logging.info('Applying clustering')
        logging.info('{} states in the graph'.format(len(self.G.nodes())))
        self.G, _, _ = apply_clustering(self.G)
        # drop not useful resources
        self.tree = None
        self.count_dict = None
        gc.collect()

    def recommend(self, user_profile, user_id=None):

        # if the user profile is longer than the markov order, chop it keeping recent history
        state = tuple(user_profile[-self.order:])
        # see if graph has that state
        recommendations = []
        if self.G.has_node(state):
            # search for recommendations in the forward star
            rec_dict = {}
            for u, v in self.G.out_edges_iter([state]):
                lastElement = tuple(v[-1:])
                if lastElement in rec_dict:
                    rec_dict[lastElement] += self.G[u][v]['count']
                else:
                    rec_dict[lastElement] = self.G[u][v]['count']
            for k, v in rec_dict.items():
                recommendations.append((list(k), v))

        return recommendations

    def _set_graph_debug(self, G):
        self.G = G
