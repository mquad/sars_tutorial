import logging

from sars_tutorial.recommenders.ISeqRecommender import ISeqRecommender
from sars_tutorial.recommenders.MarkovChainRecommender import MarkovChainRecommender


class MixedMarkovChainRecommender(ISeqRecommender):
    """
    Creates markov models with different values of k, and return recommendation by weighting the list of
    recommendation of each model.

    Reference: Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
    Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    recommenders = {}

    def __init__(self, min_order=1, max_order=1):
        """
        :param min_order: the minimum order of the Mixed Markov Chain
        :param max_order: the maximum order of the Mixed Markov Chain
        """
        super(MixedMarkovChainRecommender, self).__init__()
        self.min_order = min_order
        self.max_order = max_order
        # define the models
        for i in range(self.min_order, self.max_order + 1):
            self.recommenders[i] = MarkovChainRecommender(i)

    def fit(self, user_profile):
        for order in self.recommenders:
            self.recommenders[order].fit(user_profile)

    def recommend(self, user_profile, user_id=None):
        rec_dict = {}
        recommendations = []
        sum_of_weights = 0
        for order, r in self.recommenders.items():
            rec_list = r.recommend(user_profile)
            sum_of_weights += 1 / order
            for i in rec_list:
                if tuple(i[0]) in rec_dict:
                    rec_dict[tuple(i[0])] += 1 / order * i[1]
                else:
                    rec_dict[tuple(i[0])] = 1 / order * i[1]
        for k, v in rec_dict.items():
            recommendations.append((list(k), v / sum_of_weights))

        return recommendations

    def _set_model_debug(self, recommender, order):
        self.recommenders[order] = recommender
