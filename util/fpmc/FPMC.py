import sys, os, pickle, time
import math, random
import numpy as np
from util.fpmc.utils import *
import logging

class FPMC():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        self.user_set = set()
        self.item_set = set()

        self.n_user = n_user
        self.n_item = n_item

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    @staticmethod
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def init_model(self, std=0.01):
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def compute_x(self, u, i, b_tm1):
        acc_val = 0.0
        for l in b_tm1:
            acc_val += np.dot(self.VIL[i], self.VLI[l])
        return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val/len(b_tm1)))

    def compute_x_batch(self, u, b_tm1):
        former = self.VUI_m_VIU[u]
        latter = np.mean(self.VIL_m_VLI[:, b_tm1], axis=1).T
        return (former + latter)

    def evaluation(self, data_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        correct_count = 0
        rr_list = []
        for (u, i, b_tm1) in data_list:
            scores = self.compute_x_batch(u, b_tm1)

            if i == scores.argmax():
                correct_count += 1

            rank = len(np.where(scores > scores[i])[0]) + 1
            rr = 1.0/rank
            rr_list.append(rr)

        try:
            acc = correct_count / len(rr_list)
            mrr = (sum(rr_list) / len(rr_list))
            return (acc, mrr)
        except:
            return (0.0, 0.0)

    def learn_epoch(self, tr_data, neg_batch_size):
        for iter_idx in range(len(tr_data)):
            (u, i, b_tm1) = random.choice(tr_data)
            
            exclu_set = self.item_set - set([i])
            j_list = random.sample(exclu_set, neg_batch_size)
            
            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:

                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)

                VUI_update = self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                VIUi_update = self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                VIUj_update = self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])

                self.VUI[u] += VUI_update
                self.VIU[i] += VIUi_update
                self.VIU[j] += VIUj_update

                eta = np.mean(self.VLI[b_tm1], axis=0)
                VILi_update = self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                VILj_update = self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                VLI_update = self.learn_rate * ((delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

                self.VIL[i] += VILi_update
                self.VIL[j] += VILj_update
                self.VLI[b_tm1] += VLI_update

    def learnSBPR_FPMC(self, tr_data, n_epoch=10, neg_batch_size=10):
        for epoch in range(n_epoch):
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)
            self.logger.info ('epoch %d done' % epoch)
            # if eval_per_epoch == True:
            #     acc_in, mrr_in = self.evaluation(tr_data)
            #     if te_data != None:
            #         acc_out, mrr_out = self.evaluation(te_data)
            #         self.logger.info ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
            #     else:
            #         self.logger.info ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))
            # else:
            #



        # if eval_per_epoch == False:
        #     acc_in, mrr_in = self.evaluation(tr_data)
        #     if te_data != None:
        #         acc_out, mrr_out = self.evaluation(te_data)
        #         print ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
        #     else:
        #         print ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))
        #
        # if te_data != None:
        #     return (acc_out, mrr_out)
        # else:
        #     return None
