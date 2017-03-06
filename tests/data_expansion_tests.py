from util.data_expansion import data_expansion,user_profile_expansion
from scipy.sparse import csc_matrix
import numpy as np
import unittest

class Data_expansion_tests(unittest.TestCase):

    sequences = [[1,2,3,4],[9,7,4],[3,2,1],[0,4,3,2]]

    def test(self):
        row_tru = [0,1,1,1,2,2,2,2,3,3,3,3,3,4,5,5,5,6,6,6,6,7,8,8,8,9,9,9,9,10,11,11,11,12,12,12,12,13,13,13,13,13]
        col_tru = [0,1,7,14,2,8,14,15,3,9,14,15,16,4,5,11,18,3,12,18,19,2,1,9,16,0,8,15,16,6,3,13,20,2,10,17,20,1,9,16,17,20]
        truth_h1 = csc_matrix((np.ones(len(row_tru)),(row_tru,col_tru)))
        assert (data_expansion(self.sequences,1)[0] != truth_h1).nnz == 0

    def test_user_profile(self):

        data,mapping = data_expansion(self.sequences,1)
        user_profile = [1,2,3]
        res  = user_profile_expansion(user_profile,1,mapping)
        row_tru = [0,0,0,0]
        col_tru = [2,7,8,9]
        truth_h1 = csc_matrix((np.ones(len(row_tru)),(row_tru,col_tru)),shape=(1,len(mapping)*(2)))
        assert (res != truth_h1).nnz ==0




