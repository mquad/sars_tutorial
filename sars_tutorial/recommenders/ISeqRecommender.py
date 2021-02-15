import logging


class ISeqRecommender(object):
    """Abstract Recommender class"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, train_data):
        pass

    def recommend(self, user_profile, user_id=None):
        """
        Given the user profile return a list of recommendation
        :param user_profile: the user profile as a list of item identifiers
        :param user_id: (optional) the user id
        :return: list of recommendations e.g. [([2], 0.875), ([6], 1.0)]
        """
        pass

    @staticmethod
    def get_recommendation_list(recommendation):
        return list(map(lambda x: x[0], recommendation))

    @staticmethod
    def get_recommendation_confidence_list(recommendation):
        return list(map(lambda x: x[1], recommendation))

    def activate_debug_print(self):
        self.logger.setLevel(logging.DEBUG)

    def deactivate_debug_print(self):
        self.logger.setLevel(logging.INFO)
