from recommenders.ISeqRecommender import ISeqRecommender
try:
    from util.fpmc.FPMC_numba import FPMC
    print ('Using numba')
except ImportError:
    from util.fpmc import FPMC

class FPMCRecommender(ISeqRecommender):

    def __init__(self,user_set,item_set,n_factor,learn_rate,regular,n_epochs,n_neg,top_n):
        super(FPMCRecommender, self).__init__()
        self.fpmc = FPMC(n_user=max(user_set)+1, n_item=max(item_set)+1,
                    n_factor=n_factor, learn_rate=learn_rate, regular=regular)
        self.fpmc.user_set = user_set
        self.fpmc.item_set = item_set
        self.fpmc.init_model()
        self.n_epochs = n_epochs
        self.n_neg = n_neg
        self.top_n=top_n

    def set_user_list(self,user_list):
        self.user_list = user_list

    def fit(self, sequences):

        acc, mrr = self.fpmc.learnSBPR_FPMC(tr_data, te_data, n_epoch=self.n_epoch,
                                   neg_batch_size=self.n_neg, eval_per_epoch=False)
