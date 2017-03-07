import math
import numpy as np
from numba import jit
from util.fpmc.utils import *
from util.fpmc import FPMC as FPMC_basic
import logging

class FPMC(FPMC_basic.FPMC):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        super(FPMC, self).__init__(n_user, n_item, n_factor, learn_rate, regular)

    def evaluation(self, data_3_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        acc, mrr = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], self.VUI_m_VIU, self.VIL_m_VLI)

        return acc, mrr

    def evaluation_recommender(self, user,user_profile):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        scores = evaluation_jit_recommender(user, user_profile, self.VUI_m_VIU, self.VIL_m_VLI)
        return sorted(range(len(scores)), key=lambda x: -scores[x]),sorted(scores,reverse=True)

    def learn_epoch(self, data_3_list, neg_batch_size):
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_3_list[0], data_3_list[1], data_3_list[2], neg_batch_size,
                                             np.array(list(self.item_set)), self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def learnSBPR_FPMC(self, tr_data, n_epoch=10, neg_batch_size=10):
        tr_3_list = data_to_3_list(tr_data)

        for epoch in range(n_epoch):
            self.learn_epoch(tr_3_list, neg_batch_size)
            self.logger.info('epoch %d done' % epoch)

        # if eval_per_epoch == False:
        #     acc_in, mrr_in = self.evaluation(tr_3_list)
        #     if te_data != None:
        #         acc_out, mrr_out = self.evaluation(te_3_list)
        #         print ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
        #     else:
        #         print ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))
        #
        #
        # if te_data != None:
        #     if ret_in_score:
        #         return (acc_in, mrr_in, acc_out, mrr_out)
        #     else:
        #         return (acc_out, mrr_out)
        # else:
        #     return None


@jit(nopython=True)
def compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL):
    acc_val = 0.0
    for l in b_tm1:
        acc_val += np.dot(VIL[i], VLI[l])
    return (np.dot(VUI[u], VIU[i]) + (acc_val/len(b_tm1)))



@jit(nopython=True)
def learn_epoch_jit(u_list, i_list, b_tm1_list, neg_batch_size, item_set, VUI, VIU, VLI, VIL, learn_rate, regular):
    for iter_idx in range(len(u_list)):
        d_idx = np.random.randint(0, len(u_list))
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx]!=-1]

        j_list = np.random.choice(item_set, size=neg_batch_size, replace=False)
        z1 = compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL)
        for j in j_list:
            z2 = compute_x_jit(u, j, b_tm1, VUI, VIU, VLI, VIL)
            delta = 1 - sigmoid_jit(z1 - z2)

            VUI_update = learn_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])
            VIUi_update = learn_rate * (delta * VUI[u] - regular * VIU[i])
            VIUj_update = learn_rate * (-delta * VUI[u] - regular * VIU[j])

            VUI[u] += VUI_update
            VIU[i] += VIUi_update
            VIU[j] += VIUj_update

            eta = np.zeros(VLI.shape[1])
            for l in b_tm1:
                eta += VLI[l]
            eta = eta / len(b_tm1)

            VILi_update = learn_rate * (delta * eta - regular * VIL[i])
            VILj_update = learn_rate * (-delta * eta - regular * VIL[j])
            VLI_updates = np.zeros((len(b_tm1), VLI.shape[1]))
            for idx, l in enumerate(b_tm1):
                VLI_updates[idx] = learn_rate * ((delta * (VIL[i] - VIL[j]) / len(b_tm1)) - regular * VLI[l])

            VIL[i] += VILi_update
            VIL[j] += VILj_update
            for idx, l in enumerate(b_tm1):
                VLI[l] += VLI_updates[idx]

    return VUI, VIU, VLI, VIL

@jit(nopython=True)
def sigmoid_jit(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))

@jit(nopython=True)
def compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI):
    former = VUI_m_VIU[u]
    latter = np.zeros(VIL_m_VLI.shape[0])
    for idx in range(VIL_m_VLI.shape[0]):
        for l in b_tm1:
            latter[idx] += VIL_m_VLI[idx, l]
    latter = latter/len(b_tm1)

    return (former + latter)

@jit(nopython=True)
def evaluation_jit(u_list, i_list, b_tm1_list, VUI_m_VIU, VIL_m_VLI):
    correct_count = 0
    acc_rr = 0
    for d_idx in range(len(u_list)):
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx]!=-1]
        scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)


        if i == scores.argmax():
            correct_count += 1

        rank = len(np.where(scores > scores[i])[0]) + 1
        rr = 1.0/rank
        acc_rr += rr

    acc = correct_count / len(u_list)
    mrr = acc_rr / len(u_list)
    return (acc, mrr)

@jit(nopython=True)
def evaluation_jit_recommender(user, b_tm1_list, VUI_m_VIU, VIL_m_VLI):

    u = user
    #b_tm1 = [x for x in b_tm1_list if x!=-1]
    b_tm1 = b_tm1_list
    scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)

    return scores
