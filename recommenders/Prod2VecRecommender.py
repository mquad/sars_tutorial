import logging

import gensim

from recommenders.ISeqRecommender import ISeqRecommender


class Prod2VecRecommender(ISeqRecommender):
    """
    Implementation of the Prod2Vec skipgram model from
    Grbovic Mihajlo, Vladan Radosavljevic, Nemanja Djuric, Narayan Bhamidipati, Jaikit Savla, Varun Bhagwan, and Doug Sharp.
    "E-commerce in your inbox: Product recommendations at scale."
    In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
    pp. 1809-1818. ACM, 2015.
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, min_count=2, size=100, window=5, decay_alpha=0.9, workers=4):
        """
        :param min_count: (optional) the minimum item frequency. Items less frequent that min_count will be pruned
        :param size: (optional) the size of the embeddings
        :param window: (optional) the size of the context window
        :param decay_alpha: (optional) the exponential decay factor used to discount the similarity scores for items
                back in the user profile. Lower values mean higher discounting of past user interactions. Allows values in [0-1].
        :param workers: (optional) the number of threads used for training
        """
        super(Prod2VecRecommender, self).__init__()
        self.min_count = min_count
        self.size = size
        self.window = window
        self.decay_alpha = decay_alpha
        self.workers = workers

    def __str__(self):
        return 'Prod2VecRecommender(min_count={min_count}, ' \
               'size={size}, ' \
               'window={window}, ' \
               'decay_alpha={decay_alpha}, ' \
               'workers={workers})'.format(**self.__dict__)

    def fit(self, train_data):
        sequences = train_data['sequence'].values
        self.model = gensim.models.Word2Vec(sequences,
                                            min_count=self.min_count,
                                            window=self.window,
                                            hs=1,
                                            size=self.size,
                                            sg=1,
                                            workers=self.workers)

    def recommend(self, user_profile, user_id=None):
        user_profile = list(map(str, user_profile))
        rec = []
        try:
            # iterate the user profile backwards
            for i, item in enumerate(user_profile[::-1]):
                ms = self.model.most_similar(positive=item)
                # apply exponential decay to the similarity scores
                decay = self.decay_alpha ** i
                ms = [(x[0], decay * x[1]) for x in ms]
                rec.extend(ms)
            # sort items by similarity score
            rec = sorted(rec, key=lambda x: -x[1])
        except KeyError:
            rec = []
        return [([x[0]], x[1]) for x in rec]
