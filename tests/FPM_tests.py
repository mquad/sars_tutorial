import ast
import pandas as pd
from recommenders.Freq_Seq_Mining_Recommender import FreqSeqMiningRecommender
from util.tree.Tree import SmartTree
from util import evaluation,metrics
import numpy as np
import unittest

class FPM_recommender_tests(unittest.TestCase):

    def build_smart_tree(self):
        t = SmartTree()
        rootNode = t.set_root()
        defaultSupport = 1
        t.add_path(rootNode,[1,2],defaultSupport)
        t.add_path(rootNode,[1,3,1],8)
        t.add_path(rootNode,[1,3,1,0],8)
        t.add_path(rootNode,[1,3,1,6],1)
        t.add_path(rootNode,[1,3,1,4],6)
        t.add_path(rootNode,[1,3,1,4,9],3)
        t.add_path(rootNode,[1,3,1,4,2],3)
        t.add_path(rootNode,[1,3,6],defaultSupport)
        t.add_path(rootNode,[1,3,2],defaultSupport)
        t.add_path(rootNode,[2,3],defaultSupport)
        t.add_path(rootNode,[2,1],defaultSupport)
        t.add_path(rootNode,[3,4,2,1,5,1],defaultSupport)
        t.add_path(rootNode,[3,4,2,6],defaultSupport)
        t.add_path(rootNode,[3,4,1,6],defaultSupport)
        t.add_path(rootNode,[3,4,1,5,5],defaultSupport)
        t.add_path(rootNode,[3,4,1,5,4],defaultSupport)
        t.add_path(rootNode,[3,5],defaultSupport)
        t.add_path(rootNode,[9],defaultSupport)
        t.add_path(rootNode,[4,2,1,5,1,3],1)
        t.add_path(rootNode,[4,2,1,5,1],8)
        t.add_path(rootNode,[4,2,1,5,1,2],7)
        t.add_path(rootNode,[4,2,1,5,1,6],8)
        t.add_path(rootNode,[4,2,1,5,1,6,9],4)
        t.add_path(rootNode,[4,2,1,5,1,6,9,3],1)
        t.add_path(rootNode,[4,2,1,5,1,6,9,0],3)
        t.add_path(rootNode,[4,2,1,5,1,6,4],4)

        return t

    def test_fit(self):

        db=pd.read_csv('datasets/test_data/simple.txt',converters={'songs':ast.literal_eval})
        seqs = db['songs'].tolist()

        rec = FreqSeqMiningRecommender(0.9,0.1,True)
        rec.fit(seqs)

        ## test that the right frequences are found
        seq_ground_truth={((1,),4),((1,2),4),((1,2,3),4),((1,3),4),((2,),4),((2,3),4),((3,),4)}
        self.assertEqual(rec.get_freq_seqs(),seq_ground_truth)

        ## test right tree build by checking the existence and support
        ## of all paths. NB actually there could be more paths than does needed
        tree = rec.get_sequence_tree()

        self.assertNotEqual(tree.find_path(tree.get_root(),[1]), -1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[1])).data['support'], 4)

        self.assertNotEqual(tree.find_path(tree.get_root(),[1,2]), -1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[1,2])).data['support'],4)

        self.assertNotEqual(tree.find_path(tree.get_root(),[1,2,3]) , -1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[1,2,3])).data['support'] ,4)

        self.assertNotEqual(tree.find_path(tree.get_root(),[1,3]) , -1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[1,3])).data['support'],4)

        self.assertNotEqual(tree.find_path(tree.get_root(),[2]) , -1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[2])).data['support'],4)

        self.assertNotEqual(tree.find_path(tree.get_root(),[2,3]), -1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[2,3])).data['support'] ,4)

        self.assertNotEqual(tree.find_path(tree.get_root(),[3]),-1)
        self.assertEqual(tree.get_node(tree.find_path(tree.get_root(),[3])).data['support'] ,4)

    def test_recommendation(self):

        t = self.build_smart_tree()

        rec1 = FreqSeqMiningRecommender(0.9,0.2,9,1)
        rec2 = FreqSeqMiningRecommender(0.9,0.2,9,6)
        rec3 = FreqSeqMiningRecommender(0.9,0.2,3,1)
        rec4 = FreqSeqMiningRecommender(0.9,0.2,4,1)
        rec1._set_tree_debug_only(t)
        rec2._set_tree_debug_only(t)
        rec3._set_tree_debug_only(t)
        rec4._set_tree_debug_only(t)

        s1 = [7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8]
        s2 = [1, 3, 1, 0, 6, 4]
        s3 = [8, 7, 6, 5, 4, 3, 2, 1]

        # if user profile empty then no recommendation
        recommendation = rec1.recommend([])
        assert rec1.get_recommendation_list(recommendation) == []

        recommendation = rec1.recommend(s1[:9])
        assert rec1.get_recommendation_list(recommendation) == [[2], [6]]
        assert sorted(rec1.get_confidence_list(recommendation)) == sorted([7 / 8, 1])

        recommendation = rec2.recommend(s1[:9])
        assert rec2.get_recommendation_list(recommendation)== []
        assert  sorted(rec2.get_confidence_list(recommendation)) == []

        recommendation = rec3.recommend(s2[:3])
        assert rec3.get_recommendation_list(recommendation) == [[0], [4]]
        assert sorted(rec3.get_confidence_list(recommendation)) == sorted([1, 6 / 8])

        recommendation =rec4.recommend(s3[:4])
        assert  rec4.get_recommendation_list(recommendation) == []
        assert sorted(rec4.get_confidence_list(recommendation)) == sorted([])

    def test_evaluation(self):

        t = self.build_smart_tree()

        rec1 = FreqSeqMiningRecommender(0.9,0.2,10,6)
        rec2 = FreqSeqMiningRecommender(0.9,0.2,10,1)
        rec3 = FreqSeqMiningRecommender(0.9,0.2,10,3)
        rec1._set_tree_debug_only(t)
        rec2._set_tree_debug_only(t)
        rec3._set_tree_debug_only(t)
        evaluation_functions=[metrics.precision,metrics.recall]

        test_seq = [7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8]

        #look-ahead =1: seqeunce
        assert np.array_equal(evaluation.evaluate_sequence(rec1, test_seq, 3, 1, evaluation_functions),np.zeros(len(evaluation_functions)))
        assert np.array_equal(evaluation.evaluate_sequence(rec2, test_seq, 3, 1, evaluation_functions),np.array([0.5,1]))

        #look ahead = 4 = set evaluation, ground truth less than 4
        assert np.array_equal(evaluation.evaluate_sequence(rec2, test_seq, 3, 4, evaluation_functions),np.array([0.5,1/3]))
        assert np.array_equal(evaluation.evaluate_sequence(rec2, test_seq, 3, 'total', evaluation_functions),np.array([0.5,1/3]))

        assert np.array_equal(evaluation.evaluate_sequence(rec2, [3,4,2,4,1,1], 3, 'total', evaluation_functions),np.array([0.5,0.5]))

        db = [[7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8],
              [7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 2],
              [3,4,2,4,1,1]]

        ##set evaluation all dataset
        assert np.array_equal(evaluation.set_evaluation(rec3, db, 3, 'total', evaluation_functions), np.array([2/3,1/2]))

        ##single sequential evaluation
        assert np.array_equal(evaluation.sequence_sequential_evaluation(rec2, [7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8], 3, evaluation_functions), np.array([1 / 6, 1 / 3]))
        assert np.array_equal(evaluation.sequence_sequential_evaluation(rec2, [7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 4, 8], 3, evaluation_functions), np.array([1 / 3, 2 / 3]))

        #sequential evaluation of db
        db=[[7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8],
            [7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 4, 8]]

        assert np.array_equal(evaluation.sequential_evaluation(rec2, db, 3, 'total', evaluation_functions),np.array([3/12,0.5]))

    def test_n_length_paths(self):

        t = self.build_smart_tree()
        ###test find_n_length_path
        excludeOrigin = False
        p0 = t.find_n_legth_paths(t.get_root(),0,excludeOrigin)
        p1 = t.find_n_legth_paths(t.get_root(),1,excludeOrigin)
        p2 = t.find_n_legth_paths(t.get_root(),2,excludeOrigin)
        p3 = t.find_n_legth_paths(t.get_root(),3,excludeOrigin)

        assert sorted(t.get_paths_tag(p0)) == sorted([['root']])
        assert sorted(t.get_paths_tag(p1)) == sorted([['root', 1], ['root', 2], ['root', 3], ['root', 4], ['root', 9]])
        assert sorted(t.get_paths_tag(p2)) == sorted([['root', 1, 2], ['root', 1, 3], ['root', 2, 3], ['root', 2, 1], ['root', 3, 4], ['root', 3, 5], ['root', 4, 2]])
        assert sorted(t.get_paths_tag(p3)) == sorted([['root', 1, 3, 1], ['root', 1, 3 ,6], ['root', 1, 3, 2], ['root', 3, 4, 2], ['root', 3, 4, 1], ['root', 4, 2, 1]])

        node1 = t.find_path(t.get_root(),[1])
        node3 = t.find_path(t.get_root(),[2,3])
        node4 = t.find_path(t.get_root(),[3,4])

        p0_bis1 = t.find_n_legth_paths(node4,0,excludeOrigin)
        p1_bis1 = t.find_n_legth_paths(node1,1,excludeOrigin)
        p1_bis2 = t.find_n_legth_paths(node3,1,excludeOrigin)
        p2_bis1 = t.find_n_legth_paths(node4,2,excludeOrigin)
        p3_bis1 = t.find_n_legth_paths(node3,3,excludeOrigin)
        p3_bis2 = t.find_n_legth_paths(node4,3,excludeOrigin)

        assert sorted(t.get_paths_tag(p0_bis1)) == sorted([[4]])
        assert sorted(t.get_paths_tag(p1_bis1)) == sorted([[1,2], [1, 3]])
        assert sorted(t.get_paths_tag(p1_bis2)) == sorted([])
        assert sorted(t.get_paths_tag(p3_bis1)) == sorted([])
        assert sorted(t.get_paths_tag(p2_bis1)) == sorted([[4,2,1], [4,2,6],[4,1,6],[4,1,5]])
        assert sorted(t.get_paths_tag(p3_bis2)) == sorted([[4,2,1,5], [4,1,5,5],[4,1,5,4]])

        ########################################################################################
        excludeOrigin = True
        p0 = t.find_n_legth_paths(t.get_root(),0,excludeOrigin)
        p1 = t.find_n_legth_paths(t.get_root(),1,excludeOrigin)
        p2 = t.find_n_legth_paths(t.get_root(),2,excludeOrigin)
        p3 = t.find_n_legth_paths(t.get_root(),3,excludeOrigin)

        assert sorted(t.get_paths_tag(p0)) == sorted([[]])
        assert sorted(t.get_paths_tag(p1)) == sorted([[1], [2], [3], [4], [9]])
        assert sorted(t.get_paths_tag(p2)) == sorted([[1, 2], [1, 3], [2, 3], [2, 1], [3, 4], [3, 5], [4, 2]])
        assert sorted(t.get_paths_tag(p3)) == sorted([[1, 3, 1], [1, 3 ,6], [1, 3, 2], [ 3, 4, 2], [3, 4, 1], [4, 2, 1]])

        node1 = t.find_path(t.get_root(),[1])
        node3 = t.find_path(t.get_root(),[2,3])
        node4 = t.find_path(t.get_root(),[3,4])

        p0_bis1 = t.find_n_legth_paths(node4,0,excludeOrigin)
        p1_bis1 = t.find_n_legth_paths(node1,1,excludeOrigin)
        p1_bis2 = t.find_n_legth_paths(node3,1,excludeOrigin)
        p2_bis1 = t.find_n_legth_paths(node4,2,excludeOrigin)
        p3_bis1 = t.find_n_legth_paths(node3,3,excludeOrigin)
        p3_bis2 = t.find_n_legth_paths(node4,3,excludeOrigin)

        assert sorted(t.get_paths_tag(p0_bis1)) == sorted([[]])
        assert sorted(t.get_paths_tag(p1_bis1)) == sorted([[2], [3]])
        assert sorted(t.get_paths_tag(p1_bis2)) == sorted([])
        assert sorted(t.get_paths_tag(p3_bis1)) == sorted([])
        assert sorted(t.get_paths_tag(p2_bis1)) == sorted([[2,1], [2,6],[1,6],[1,5]])
        assert sorted(t.get_paths_tag(p3_bis2)) == sorted([[2,1,5], [1,5,5],[1,5,4]])