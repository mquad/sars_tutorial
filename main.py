import logging
from collections import OrderedDict
from recommenders import FreqSeqMining
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('freq_seqs', FreqSeqMining)
])

##laod data
import ast
db=pd.read_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()


#
# import networkx as nx
# import matplotlib.pyplot as plt
# G=nx.DiGraph()
# G.add_edges_from(
#     [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
#      ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])
# pos=nx.spring_layout(G)
# nx.draw(G,pos=pos)
# nx.draw_networkx_labels(G,pos=pos)
# plt.show()
#
# tree=nx.DiGraph()
#
# ##add root
# rootNode = 'root'
# tree.add_node(rootNode)
#
# for tuple in freq_seqs:
#     if len(tuple[0]) == 1:
#         #add node to root
#         tree.add_edge(rootNode,tuple[0][0])
#     elif len(tuple[0]) >1:
#         #add entire path starting from root
#         tree.add_path((rootNode,) + tuple[0])
#     else:
#         logger.critical('Error frequent sequence of lenght 0')
#
# pos=nx.spring_layout(tree)
# nx.draw(tree,pos=pos)
# nx.draw_networkx_labels(tree,pos=pos)
# plt.show()
#
# ##
#
# nx.is_directed_acyclic_graph(tree)
#



# tree.create_node(0, rootNode)
# tree.create_node('1',1,parent='root')
# tree.create_node('2',2,parent=1,data={'support':10})
# tree.create_node('2',3,parent='root',data={'support':0})
# tree.create_node('3',4,parent=2,data={'support':0})
# tree.create_node('4',5,parent=3,data={'support':0})
#
