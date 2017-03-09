from recommenders.ISeqRecommender import ISeqRecommender
import operator

class PopularityRecommender(ISeqRecommender):

    def __init__(self):
        super(PopularityRecommender, self).__init__()

    def fit(self, sequences):
        count_dict={}
        for s in sequences:
            for item in s:
                if item not in count_dict:
                    count_dict[item]=1
                else:
                    count_dict[item]+=1

        self.top = sorted(count_dict.items(), key=operator.itemgetter(1),reverse=True)
        self.top = [([x[0]],x[1]) for x in self.top]

    def recommend(self, user_profile):
        return self.top

    def get_popular_list(self):
        return self.top