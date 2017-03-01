import logging
class ISeqRecommender(object):
    """Abstract Recommender"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, sequences):
        pass

    def recommend(self, user_profile):
        pass


    def get_recommendation_list(self,recommendation):
        return list(map(lambda x:x[0],recommendation))

    def get_recommendation_confidence_list(self,recommendation):
        return list(map(lambda x:x[1],recommendation))

    def activate_debug_print(self):
        self.logger.setLevel(logging.DEBUG)

    def deactivate_debug_print(self):
        self.logger.setLevel(logging.INFO)

