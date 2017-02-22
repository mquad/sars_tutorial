import pandas as pd
import ast
from recommenders.FreqSeqMining import FreqSeqMiningRecommender
from tree.Tree import SmartTree

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

####testing recommendation of FPM

## manually create tree

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
t.add_path(rootNode,[3,4,2,5,1],defaultSupport)
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
t.add_path(rootNode,[4,2,1,5,1,6,9,3],1)
t.add_path(rootNode,[4,2,1,5,1,6,9,0],3)
t.add_path(rootNode,[4,2,1,5,1,6,4],4)


rec = FreqSeqMiningRecommender(0.9,0.2,True)
rec._set_tree_debug_only(t)

raccomandation = rec.recommend([7, 8, 9, 3, 4, 2, 1, 5, 1, 6, 7, 8], 9, 1, 1)
assert set(map(lambda x:x[0], raccomandation)) == set([2, 6])
assert sorted(list(map(lambda x:x[1], raccomandation))) == sorted([7 / 8, 1])

raccomandation = rec.recommend([7,8,9,3,4,2,1,5,1,6,7,8],9,6,1)
assert set(map(lambda x:x[0], raccomandation)) == set([])
assert sorted(list(map(lambda x:x[1], raccomandation))) == []

raccomandation = rec.recommend([1,3,1,0,6,4],3,1,1)
assert set(map(lambda x:x[0], raccomandation)) == set([0,4])
assert sorted(list(map(lambda x:x[1], raccomandation))) == sorted([1,6/8])

raccomandation =rec.recommend([8,7,6,5,4,3,2,1],4,1,1)
assert set(map(lambda x:x[0], raccomandation)) == set([])
assert sorted(list(map(lambda x:x[1], raccomandation))) == sorted([])
