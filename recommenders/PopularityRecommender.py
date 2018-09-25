import operator

from recommenders.ISeqRecommender import ISeqRecommender


class PopularityRecommender(ISeqRecommender):

    def __init__(self):
        super(PopularityRecommender, self).__init__()

    def fit(self, train_data):
        sequences = train_data['sequence'].values

        count_dict = {}
        for s in sequences:
            for item in s:
                if item not in count_dict:
                    count_dict[item] = 1
                else:
                    count_dict[item] += 1

        self.top = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.top = [([x[0]], x[1]) for x in self.top]

    def recommend(self, user_profile, user_id=None):
        return self.top

    def get_popular_list(self):
        return self.top
