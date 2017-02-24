import pandas as pd
import ast
import networkx as nx
from util.tree.Tree import SmartTree
from functools import reduce

db=pd.read_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()[:100]

t = SmartTree()
rootNode = t.set_root()

last_k = 5
countDict = {}
G=nx.DiGraph()
for i, s in enumerate(seqs):
    print(i)
    nearHistory = tuple(s[-(last_k):])
    if nearHistory in countDict:
        #increment count
        countDict[nearHistory] += 1
    else:
        #init count
        countDict[nearHistory] = 1
        #add seq to sequence tree
        t.add_path(rootNode,list(nearHistory))
        #add node to graph
        G.add_node(nearHistory)

    ## i also have to save the seuqence of length k+1 because otherwise I cannot calculate the count
    ## from state x to state y. So the seqeunces of length k+1 are in the tree but not in the states
    nearHistoryLong = tuple(s[-(last_k+1):])# +1 because I need one more element to calculate the transition prob
    if nearHistoryLong in countDict:
        #increment count
        countDict[nearHistoryLong]+= 1
    else:
        #init count
        countDict[nearHistoryLong] = 1

#uniqueSeqs = sorted(uniqueSeqs,key=lambda x: len(x))

#add links
for node in G.nodes_iter():
    # if the sequence is shorter than states's len, the next state has all the sequence as prefix
    next_state_prefix = node[1:] if len (node) == last_k else node
    p = t.find_path(rootNode,next_state_prefix)
    if t.path_is_valid(p):
        children = t.get_nodes_tag(t[p].fpointer)
        for c in children:
            G.add_edge(node,node[1:]+(c,),{'count':countDict.get(node+(c,),0)})
    else:
        continue #no edge to add

# iterate over seqs to add skipping count
window = 3

for us in seqs:
    s=tuple(us)
    for i in range(len(s)-window):
        previous_state = s[i:i+window]
        next_state_prefix = previous_state[1:]
        for j in range(i+3,len(s)):
            fractional_count = 1/(2**(j-(i+3)))
            next_state = next_state_prefix+(s[j],)
            #update count
            old_count = G.get_edge_data(previous_state,next_state,{}).get('count',0)
            if G.has_edge(previous_state,next_state):
                G[previous_state][next_state]['count'] = old_count + fractional_count
            else:
                G.add_edge(previous_state,next_state,{'count': fractional_count})
            #print('updating '+str(previous_state)+'->'+str(next_state)+' from '+str(old_count)+' to '+str(old_count+fractional_count))

#normalize
for n in G.nodes_iter():
    edges=G.out_edges(n)
    countSum = reduce(lambda x,y:x+y,[G[x[0]][x[1]]['count'] for x in edges],0)
    for e in edges:
        G[e[0]][e[1]]['count'] =  G[e[0]][e[1]]['count']/float(countSum) if countSum else 0

