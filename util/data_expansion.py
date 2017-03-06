from scipy.sparse import csc_matrix
import numpy as np


def data_expansion(sequences,history_length):
    # sequences = [[1,2,3,4],[9,7,4],[3,2,1],[0,4,3,2]]
    # history_length = 3

    #store unique elements
    #mapping items to incremental integers

    count=0
    items_mapping = {}
    for s in sequences:
        for i in s:
            if i in items_mapping: continue
            items_mapping[i]=count
            count+=1

    number_of_unique_items = len(items_mapping)

    row = 0
    row_indeces=[]
    col_indeces=[]
    #for each sequence
    for s in sequences:
        #for each item in the sequence
        cached = []
        for i,item in enumerate(s):
            index = items_mapping[item]

            #in each row there will be: the taget,the cache
            row_indeces += [row]*( 1 + len(cached))

            #add data target
            col_indeces.append(index)

            #add history
            for l in range(1,history_length+1):
                if i < l: continue #no history available that far
                row_indeces.append(row)
                l_th_previous_item = s[i-l]
                previous_el_index = items_mapping[l_th_previous_item]
                col_indeces.append(previous_el_index + number_of_unique_items*l)

            #add cache
            col_indeces += cached
            cached.append(index + number_of_unique_items * (history_length+1))
            assert len(row_indeces) == len(col_indeces)

            row+=1

    return csc_matrix((np.ones(len(row_indeces),dtype=np.int8),(row_indeces,col_indeces)),shape=(row,(history_length + 2)*len(items_mapping))),items_mapping


def user_profile_expansion(user_profile,history_length,items_mapping):

    number_of_unique_items = len(items_mapping)

    row_indeces=[]
    col_indeces=[]

    #for each item in the sequence
    cached = [items_mapping[x]+number_of_unique_items * (history_length) for x in user_profile]
    last = user_profile[len(user_profile)-1]
    index = items_mapping[last]

    #in each row there will be:the cache
    row_indeces += [0]*(len(cached))

    #add history
    for l in range(1,history_length+1):
        if len(user_profile) < l: continue #no history available that far
        row_indeces.append(0)
        l_th_previous_item = user_profile[len(user_profile)-l]
        previous_el_index = items_mapping[l_th_previous_item]
        col_indeces.append(previous_el_index + number_of_unique_items * (l-1))

    #add cache
    col_indeces += cached

    assert len(row_indeces) == len(col_indeces)

    return csc_matrix((np.ones(len(row_indeces)),(row_indeces,col_indeces)),shape=(1,(history_length + 1)*len(items_mapping)))
