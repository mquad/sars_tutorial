from pymining import seqmining
from util.SPMFinterface import callSPMF
from recommenders.ISeqRecommender import ISeqRecommender
from util.tree.Tree import SmartTree


class FreqSeqMiningRecommender(ISeqRecommender):
    """Frequent sequence mining recommender"""

    outputPath = "tmp_output.txt"

    def __init__(self, minsup, minconf, max_context, min_context=1, spmf_path=None, db_path=None):
        """minsup is interpreted as percetage if [0-1] or as count if > 1.
        spmf_path is the path where the spmf jar is while db_path is the path of the sequence db
        in spmf format. Both have to be valid in order to use spfm for sequence mining"""

        super(FreqSeqMiningRecommender, self).__init__()
        self.minsup = minsup
        self.minconf = minconf
        self.max_context = max_context
        self.min_context = min_context
        self.recommendation_length = 1
        self.spmf_path = spmf_path
        self.db_path = db_path

    def fit(self,seqs):
        """Takes a list of list of sequences ."""

        if self.spmf_path and self.db_path:
            self.logger.info("Using SPMF")
            #parse minsup
            if 0 <= self.minsup <=1:
                percentage_min_sup = self.minsup * 100
            else: raise NameError("SPMF only accepts 0<=minsup<=1")

            #call spmf
            algorithm = "PrefixSpan"
            command = ' '.join([algorithm, self.db_path, self.outputPath, str(percentage_min_sup) + '%'])
            callSPMF(self.spmf_path, command)

            #parse back output from text file
            self._parse_SPMF_output()
        elif seqs:
            msup = self.minsup * len(seqs) if 0 <= self.minsup <=1 else self.minsup

            self.logger.debug('Mining frequent sequences')
            self.freq_seqs = seqmining.freq_seq_enum(seqs, msup)
        else:
            self.logger.error("No sequence dabase path nor sequence list provided.")

        self.logger.info('{} frequent sequences found'.format(len(self.freq_seqs)))
        self.logger.debug('Building frequent sequence tree')
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
        self.logger.debug('Tree completed')

    def recommend(self, user_profile):
        '''
        Given the user profile return a list of recomendation
        :param user_profile: user profile
        :return: list of recommendations e.g. [([2], 0.875), ([6], 1.0)]
        '''
        n = len(user_profile)
        c = min(n, self.max_context)
        match = []
        while not match and c >= self.min_context:
            q = user_profile[n-c:n]
            match = self._find_match(q,self.recommendation_length)
            c -= 1
        return match

    def _find_match(self,context,recommendation_length):
        #self.logger.debug('Searching match '+str(context))

        #search context
        lastNode = self.tree.find_path(self.rootNode,context)

        if lastNode == -1:
            #self.logger.debug('Context match not found')
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

    def get_confidence_list(self,recommendation):
        return list(map(lambda x:x[1],recommendation))

    def _parse_SPMF_output(self):
        with open(self.outputPath,'r') as fin:
            self.freq_seqs = []
            for line in fin:
                pieces = line.split('#SUP: ')
                support = pieces[1].strip()
                items = pieces[0].split(' ')
                seq = tuple(x for x in items if x!='' and x!='-1')
                seq_and_support = ((seq,int(support)))
                self.freq_seqs.append(seq_and_support)
