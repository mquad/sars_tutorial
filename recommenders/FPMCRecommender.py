from recommenders.ISeqRecommender import ISeqRecommender
from util.fpmc.FPMC_numba import FPMC


class FPMCRecommender(ISeqRecommender):
    """
    Implementation of
    Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010). Factorizing personalized Markov chains for next-basket recommendation.
    Proceedings of the 19th International Conference on World Wide Web - WWW ’10, 811

    Based on the implementation available at https://github.com/khesui/FPMC
    """

    def __init__(self, n_factor=32, learn_rate=0.01, regular=0.001, n_epoch=15, n_neg=10):
        """
        :param n_factor: (optional) the number of latent factors
        :param learn_rate: (optional) the learning rate
        :param regular: (optional) the L2 regularization coefficient
        :param n_epoch: (optional) the number of training epochs
        :param n_neg: (optional) the number of negative samples used in BPR learning
        """
        super(FPMCRecommender, self).__init__()
        self.n_epoch = n_epoch
        self.n_neg = n_neg
        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    def __str__(self):
        return 'FPMCRecommender(n_epoch={n_epoch}, ' \
               'n_neg={n_neg}, ' \
               'n_factor={n_factor}, ' \
               'learn_rate={learn_rate}, ' \
               'regular={regular})'.format(**self.__dict__)

    def fit(self, train_data):
        self._declare(train_data)

        train_data_supervised = []

        for i, row in train_data.iterrows():
            u = self.user_mapping[row['user_id']]

            seq = []
            if len(row['sequence']) > 1:  # cannot use sequences with length 1 for supervised learning
                for item in row['sequence']:
                    i = self.item_mapping[item]
                    seq.append(i)

                train_data_supervised.append((u, seq[len(seq) - 1], seq[:len(seq) - 1]))

        self.fpmc = FPMC(n_user=len(self.user_mapping), n_item=len(self.item_mapping),
                         n_factor=self.n_factor, learn_rate=self.learn_rate, regular=self.regular)

        self.fpmc.user_set = set(self.user_mapping.values())
        self.fpmc.item_set = set(self.item_mapping.values())
        self.fpmc.init_model()

        self.fpmc.learnSBPR_FPMC(train_data_supervised, n_epoch=self.n_epoch, neg_batch_size=self.n_neg)

    def recommend(self, user_profile, user_id=None):
        context = []
        for item in user_profile:
            context.append(self.item_mapping[item])

        items, scores = self.fpmc.evaluation_recommender(self.user_mapping[user_id], context)
        recommendations = []

        for i, it in enumerate(items):
            recommendations.append(([self.reverse_item_mapping[it]], scores[i]))
        return recommendations

    def _declare(self, data):
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}

        user_counter = 0
        item_counter = 0
        for i, row in data.iterrows():
            if row['user_id'] not in self.user_mapping:
                self.user_mapping[row['user_id']] = user_counter
                user_counter += 1

            for item in row['sequence']:
                if item not in self.item_mapping:
                    self.item_mapping[item] = item_counter
                    self.reverse_item_mapping[item_counter] = item
                    item_counter += 1
