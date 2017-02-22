from recommenders.ISeqRecommender import ISeqRecommender
from pymining import seqmining
from tree.Tree import SmartTree
import logging


class FreqSeqMiningRecommender(ISeqRecommender):
    """Frequent sequence mining recommender"""

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self,minsup,minconf,verbose=False):
        """minsup is interpreted as percetage if [0-1] or as count if > 1 """

        super(FreqSeqMiningRecommender, self).__init__()
        logging.basicConfig(level=logging.DEBUG) if verbose else logging.basicConfig(level=logging.WARNING)
        self.minsup = minsup
        self.minconf = minconf

    def fit(self,seqs):
        """Takes a list of list of seqeunces ."""

        msup = self.minsup * len(seqs) if 0 <= self.minsup <=1 else self.minsup

        logging.debug('Mining frequent sequences')
        self.freq_seqs = seqmining.freq_seq_enum(seqs, msup)
        logging.debug('{} frequent sequences found'.format(len(self.freq_seqs)))

        logging.debug('Building frequent sequence tree')
        self.tree = SmartTree()
        self.rootNode = self.tree.set_root()
        for tuple in self.freq_seqs:
            if len(tuple[0]) == 1:
                #add node to root
                self.tree.create_node(tuple[0][0],parent=self.rootNode,data={"support":tuple[1]})
            elif len(tuple[0]) > 1:
                #add entire path starting from root
                self.tree.add_path(self.rootNode,tuple[0],tuple[1])
            else:
                raise NameError('Frequent sequence of length 0')
        logging.debug('Tree completed')

    def recommend(self,current_session,max_win_size,min_context=1,recommendation_length=1):
        n = len(current_session)
        c = min(n,max_win_size)
        match = []
        i = 0
        while not match and i < c - min_context:
            q = current_session[i:c]
            match = self._find_match(q,recommendation_length)
            i += 1
        return match

    def _find_match(self,context,recommendation_length):
        logging.debug('Searching match '+str(context))

        #search context
        lastNode = self.tree.find_path(self.rootNode,context)

        if lastNode == -1:
            logging.debug('Context match not found')
            return []
        else: #context matched
            context_support = self.tree[lastNode].data['support']
            children = self.tree[lastNode].fpointer

            if not children: return []

            #find all path of length recommendation_length from match
            paths = self.tree.find_n_legth_paths(lastNode,recommendation_length)
            return self._filter_confidence(context_support,paths)

    def _filter_confidence(self,context_support,pathsList):
        goodPaths = []
        for p in pathsList:
            confidence = self.tree[p[len(p)-1]].data['support'] / float(context_support)
            if confidence >= self.minconf:
                goodPaths.append((self.tree.get_nodes_tag(p),confidence))
        return goodPaths

    def _set_tree_debug_only(self,tree):
        self.tree = tree
        self.rootNode = tree.get_root()

    def get_freq_seqs(self):
        return self.freq_seqs

    def get_sequence_tree(self):
        return self.tree

    def show_tree(self):
        self.tree.show()

    def get_recommendation_list(self,recommendation):
        return list(map(lambda x:x[0],recommendation))

    def get_confidence_list(self,recommendation):
        return list(map(lambda x:x[1],recommendation))
