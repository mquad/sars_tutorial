import networkx as nx
from util.tree.Tree import SmartTree
from functools import reduce
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def add_nodes_to_graph(seqs,last_k):
    t = SmartTree()
    rootNode = t.set_root()

    countDict = {}
    G=nx.DiGraph()
    for s in seqs:
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

        ## i also have to save the sequence of length k+1 because otherwise I cannot calculate the count
        ## from state x to state y. So the seqeunces of length k+1 are in the tree but not in the states
        nearHistoryLong = tuple(s[-(last_k+1):])# +1 because I need one more element to calculate the transition prob
        if nearHistory != nearHistoryLong: # otherwise short seq are counted double
            if nearHistoryLong in countDict:
                #increment count
                countDict[nearHistoryLong]+= 1
            else:
                #init count
                countDict[nearHistoryLong] = 1
    return (t,countDict,G)


def add_edges(t,countDict,G,last_k):
    """
    :param t: Tree of the sequnces available as states
    :param countDict: dicionary counting the occurence for each sequence
    :param G: the graph containing the states (each one is a sequence)
    :param last_k: the number of recent item considered
    :return: the same graph G, with edges connecting states
    """
    #add links
    rootNode = t.get_root()
    for node in G.nodes_iter():
        # if the sequence is shorter than states's len, the next state has all the sequence as prefix
        next_state_prefix = node[1:] if len (node) == last_k else node
        p = t.find_path(rootNode,next_state_prefix)
        if t.path_is_valid(p):
            children = t.get_nodes_tag(t[p].fpointer)
            for c in children:
                # the tree may suggest a children which is not a state of the graph, because it was part of a longer
                # sequence, in that case no edge has to be added
                if next_state_prefix+(c,) in G.nodes():
                    if countDict.get(node+(c,),0) != 0: # do not add edge if count is 0
                        G.add_edge(node,next_state_prefix+(c,),{'count':countDict.get(node+(c,),0)})
    return G

def apply_skipping(G, last_k, seqs):

    # iterate over seqs to add skipping count
    window = last_k

    for us in seqs:
        s=tuple(us)
        for i in range(len(s)-window):
            previous_state = s[i:i+window]
            next_state_prefix = previous_state[1:]
            for j in range(i + window + 1,len(s)):
                fractional_count = 1/(2**(j-(i+window)))
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

    return G

def apply_clustering(G):

    ##clustering
    def sequence_similarity(s,t):
        sum = 0
        for i in range(min(len(s),len(t))):
             sum += 0 if s[i] != t[i] else (i+2)
        return sum

    similarity_dict = {}
    #for each state in the graph, calculate similarity
    for node in G.nodes_iter():
        for deno in G.nodes_iter():
            if node == deno or (node,deno) in similarity_dict: continue #skip if same or already done
            else:
                sim = sequence_similarity(node,deno)
                if sim: #save only if different from zero
                    similarity_dict[node,deno] = similarity_dict[deno,node] = sim

    similarity_count_dict ={}

    for node in G.nodes_iter():
        for deno in G.nodes_iter():
            if node == deno : continue
            sum = 0
            for in_edge in G.in_edges_iter([deno]):
                intermediate_node = in_edge[0]
                if intermediate_node != node: # I want to count the effect of going through Other nodes
                    sum += similarity_dict.get((node,intermediate_node),0) * G[intermediate_node][deno]['count']
            if sum:
                similarity_count_dict[node,deno] = sum

    def compute_normalization_similarity_count(G,node):
        normalization_sum = 0
        for other_state in G.nodes_iter():
            #skip similarity with myself is 0 because of how similarity_dict is calculated
            normalization_sum += similarity_count_dict.get((node, other_state),0)
        return normalization_sum

    ##update transition probability
    ### this can be made faster(?) if I store the adjancency matrix where node are connected if
    # there is a probability due to the clustering (i.e. there is an entry in similarity_count_dict
    # in this way I only have to check those edges. now it's already pretty optimized anyway
    ALPHA = 0.5
    for node in G.nodes_iter():
        normalization_sum = compute_normalization_similarity_count(G,node)

        #first half the original transition prob
        for u,v in G.out_edges_iter([node]):
            G[u][v]['count'] *= ALPHA

        #if there is similarity probability somewhere
        if normalization_sum:
            #add similarity probability
            for deno in G.nodes_iter():
                #skip if same node or there is nothing that can be added to that node
                if node == deno or similarity_count_dict.get((node,deno),0) == 0:continue

                partial_prob = (1-ALPHA) * similarity_count_dict.get((node,deno),0) / normalization_sum

                if G.has_edge(node,deno):
                    G[node][deno]['count'] +=  partial_prob
                elif partial_prob: #there wasn't an edge but now there is partial prob from other nodes
                    G.add_edge(node,deno,{'count':partial_prob})

    return G,similarity_dict,similarity_count_dict