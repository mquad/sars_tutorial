import ast

import pandas as pd

from recommenders.FreqSeqMining import FreqSeqMiningRecommender
from util.tree.Tree import SmartTree


def buildSmartTree():
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

def testFit():

    db=pd.read_csv('datasets/test_data/simple.txt',converters={'songs':ast.literal_eval})
    seqs = db['songs'].tolist()

    rec = FreqSeqMiningRecommender(0.9,0.1,True)
    rec.fit(seqs)

    ## test that the right frequences are found
    seq_ground_truth={((1,),4),((1,2),4),((1,2,3),4),((1,3),4),((2,),4),((2,3),4),((3,),4)}
    assert rec.get_freq_seqs()==seq_ground_truth

    ## test right tree build by checking the existence and support
    ## of all paths. NB actually there could be more paths than does needed
    tree = rec.get_sequence_tree()

    assert tree.find_path(tree.get_root(),[1]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[1])).data['support'] == 4

    assert tree.find_path(tree.get_root(),[1,2]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[1,2])).data['support'] == 4

    assert tree.find_path(tree.get_root(),[1,2,3]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[1,2,3])).data['support'] == 4

    assert tree.find_path(tree.get_root(),[1,3]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[1,3])).data['support'] == 4

    assert tree.find_path(tree.get_root(),[2]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[2])).data['support'] == 4

    assert tree.find_path(tree.get_root(),[2,3]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[2,3])).data['support'] == 4

    assert tree.find_path(tree.get_root(),[3]) != -1
    assert tree.get_node(tree.find_path(tree.get_root(),[3])).data['support'] == 4

def testRecommendation():

    t = buildSmartTree()

    rec = FreqSeqMiningRecommender(0.9,0.2,True)
    rec._set_tree_debug_only(t)

    recommendation = rec.recommend([7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8], 9, 1, 1)
    assert rec.get_recommendation_list(recommendation) == [[2], [6]]
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([7 / 8, 1])

    recommendation = rec.recommend([7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8], 9, 6, 1)
    assert rec.get_recommendation_list(recommendation)== []
    assert  sorted(rec.get_confidence_list(recommendation)) == []

    recommendation = rec.recommend([1, 3, 1, 0, 6, 4], 3, 1, 1)
    assert rec.get_recommendation_list(recommendation) == [[0], [4]]
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([1, 6 / 8])

    recommendation =rec.recommend([8, 7, 6, 5, 4, 3, 2, 1], 4, 1, 1)
    assert  rec.get_recommendation_list(recommendation) == []
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([])


    ##find recommendation length n
    recommendation = rec.recommend([7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8], 9, 1, 2)
    assert  rec.get_recommendation_list(recommendation) == [[6, 9], [6, 4]]
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([0.5,0.5])

    recommendation = rec.recommend([7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8], 9, 1, 3)
    assert  rec.get_recommendation_list(recommendation) == [[6, 9, 0]]
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([3/8])

    recommendation = rec.recommend([9,0,1,3,1,6,9,5,1,2], 5, 1, 2)
    assert  rec.get_recommendation_list(recommendation) == [[4, 9], [4, 2]]
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([3/8,3/8])

    recommendation = rec.recommend([9,0,1,3,1,6,9,5,1,2], 5, 1, 4)
    assert  rec.get_recommendation_list(recommendation) == []
    assert sorted(rec.get_confidence_list(recommendation)) == sorted([])


def test_n_length_paths():

    t = buildSmartTree()
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

testFit()
test_n_length_paths()
testRecommendation()