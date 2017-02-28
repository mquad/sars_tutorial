
class ISeqRecommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, sequences):
        pass

    def recommend(self, user_profile):
        pass


    def get_recommendation_list(self,recommendation):
        return list(map(lambda x:x[0],recommendation))

