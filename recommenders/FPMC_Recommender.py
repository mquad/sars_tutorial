from recommenders.ISeqRecommender import ISeqRecommender
# try:
from util.fpmc.FPMC_numba import FPMC
#     print ('Using numba')
# except ImportError:
#     from util.fpmc.FPMC import FPMC

class FPMCRecommender(ISeqRecommender):

    def __init__(self,n_factor=32,learn_rate=0.01,regular=0.001,n_epoch=15,n_neg=10):
        super(FPMCRecommender, self).__init__()
        self.n_epoch = n_epoch
        self.n_neg = n_neg
        self.n_factor=n_factor
        self.learn_rate=learn_rate
        self.regular = regular


    def fit(self, train_data):

        train_data_supervised = []

        for i,row in train_data.iterrows():
            u = self.user_mapping[row['user_id']]

            seq=[]
            for item in row['sequence']:
                i = self.item_mapping[item]
                seq.append(i)

            train_data_supervised.append((u,seq[len(seq)-1],seq[:len(seq)-1]))


        self.fpmc = FPMC(n_user=len(self.user_mapping), n_item=len(self.item_mapping),
                    n_factor=self.n_factor, learn_rate=self.learn_rate, regular=self.regular)

        self.fpmc.user_set = set(self.user_mapping.values())
        self.fpmc.item_set = set(self.item_mapping.values())
        self.fpmc.init_model()

        self.fpmc.learnSBPR_FPMC(train_data_supervised, n_epoch=self.n_epoch,neg_batch_size=self.n_neg)


    def recommend(self, user_profile, user_id):

        context=[]
        for item in user_profile:
            context.append(self.item_mapping[item])

        items,scores = self.fpmc.evaluation_recommender(self.user_mapping[user_id],context)
        recommendations=[]

        for i,it in enumerate(items):
            recommendations.append(([self.reverse_item_mapping[it]],scores[i]))
        return recommendations

    def declare(self,data):
        '''
        Takes the dataset and collects user_id and items_id
        :param data: All the data
        '''

        self.user_mapping ={}
        self.item_mapping={}
        self.reverse_item_mapping={}

        user_counter = 0
        item_counter = 0
        for i,row in data.iterrows():
            if row['user_id'] not in self.user_mapping:
                self.user_mapping[row['user_id']]=user_counter
                user_counter+=1

            for item in row['sequence']:
                if item not in self.item_mapping:
                    self.item_mapping[item]=item_counter
                    self.reverse_item_mapping[item_counter]=item
                    item_counter+=1



