
class ISeqRecommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, X):
        pass


    def recommendSet(self, user_id,l=None, n=None):
        """Recommend a set of items for user_id, using l historic information from the user profile and returning a list
        of n recommendation."""
        pass

    def recommendSequence(self, user_id,l=None, n=None):
        """Recommend a sequence of items for user_id, using l historic information from the user profile and returning a list
        of n (ordered) recommendation."""
        pass
