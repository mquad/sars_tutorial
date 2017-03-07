from recommenders.ISeqRecommender import ISeqRecommender
try:
    from util.fpmc.FPMC_numba import FPMC
    print ('Using numba')
except ImportError:
    from util.fpmc.FPMC import FPMC

class FPMCRecommender(ISeqRecommender):

    def __init__(self,n_factor=32,learn_rate=0.01,regular=0.001,n_epoch=15,n_neg=10,top_n=5):
        super(FPMCRecommender, self).__init__()
        self.n_epoch = n_epoch
        self.n_neg = n_neg
        self.top_n=top_n
        self.n_factor=n_factor
        self.learn_rate=learn_rate
        self.regular = regular


    def fit(self, train_data):
        '''
        :param train_data: a list of tuples. Each tuple is (user_id,last_item in sequence, [previous items])
        :return:
        '''

        self.user_mapping ={}
        self.item_mapping={}
        train_data_supervised = []

        user_counter = 0
        item_counter = 0
        for i,row in train_data.iterrows():
            if row['user_id'] in self.user_mapping:
                u = self.user_mapping[int(row['user_id'])]
            else:
                self.user_mapping[row['user_id']]=user_counter
                u=user_counter
                user_counter+=1

            seq=[]
            for item in row['sequence']:
                if item in self.item_mapping:
                    i = self.item_mapping[item]
                else:
                    self.item_mapping[item]=item_counter
                    i=item_counter
                    item_counter+=1
                seq.append(i)

            train_data_supervised.append((u,seq[len(seq)-1],seq[:len(seq)-1]))


        self.fpmc = FPMC(n_user=len(self.user_mapping), n_item=len(self.item_mapping),
                    n_factor=self.n_factor, learn_rate=self.learn_rate, regular=self.regular)
        #print(self.user_mapping)
        #print(self.item_mapping)
        #print(train_data_supervised)
        self.fpmc.user_set = set(self.user_mapping.values())
        self.fpmc.item_set = set(self.item_mapping.values())
        self.fpmc.init_model()

        self.fpmc.learnSBPR_FPMC(train_data_supervised, n_epoch=self.n_epoch,neg_batch_size=self.n_neg)

