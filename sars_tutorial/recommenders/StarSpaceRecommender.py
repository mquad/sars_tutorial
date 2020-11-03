from typing import List

from subprocess import Popen, PIPE, STDOUT
from sars_tutorial.recommenders.ISeqRecommender import ISeqRecommender
from tqdm import tqdm
import sys
import os
from gensim.models import KeyedVectors
from pathlib import Path
import numpy as np
from sars_tutorial.util.data_utils import WithMetadata


# NOTE: Sparspace must be linked to a bin folder in the home env
STARSPACE = "{}/bin/starspace".format(os.environ['HOME'])


class StarSpaceRecommender(ISeqRecommender):
    interactions_fpath = "tmp/starspace_interactions.txt"
    label = "page"
    model_fpath = "tmp/starspace.model"
    model = None

    def __init__(self, min_count=2, size=100, window=5, decay_alpha=0.9, workers=4, epoch=5):
        super(StarSpaceRecommender, self).__init__()
        self.min_count = min_count
        self.size = size
        self.window = window
        self.decay_alpha = decay_alpha
        self.workers = workers
        self.epoch = epoch

    def load_keyed_vectors(self):
        self.logger.info("Loading Keyed Vectors")
        self.model = KeyedVectors.load_word2vec_format(self.model_fpath + ".txt")

    @staticmethod
    def file_len(fpath):
        return len(open(fpath).readlines())

    def log_process(self, process):
        with process.stdout:
            for lines in process.stdout:
                for line in lines.split(b'\r'):
                    self.logger.info(line.decode('utf-8').strip())
            exitcode = process.wait()
            if exitcode != 0:
                sys.exit(1)

    def run_cmd(self, arg_list):
        process = Popen(
            arg_list,
            stdout=PIPE, stderr=STDOUT
        )
        self.log_process(process)

    def convert_to_wv_format(self):
        self.logger.info("Converting to Word2Vec format.")
        tsv_file = self.model_fpath + ".tsv"
        txt_file = self.model_fpath + ".txt"
        n_vec = self.file_len(tsv_file)
        header = "{} {}\n".format(n_vec, self.size)
        with open(txt_file, "w") as out_file:
            out_file.write(header)
            with open(tsv_file) as in_file:
                for l in tqdm(in_file):
                    out_file.write(l.replace("\t", " "))

    def train_starspace(self):
        arg_list = [
             STARSPACE, "train",
             "-trainFile", self.interactions_fpath,
             "-model", self.model_fpath,
             "-label", self.label,
             "-loss", "softmax",  # default is hinge, but that is closer to word2vec
             "-trainMode", "1",
             "-epoch", str(self.epoch),
             "-minCount", str(self.min_count),
             "-dim", str(self.size),
             "-thread", str(int(self.workers * 2)),
             "-maxNegSamples", "5"  # to make star space similar to word2vec
             ]
        self.run_cmd(arg_list)

    def convert_to_starspace_input(self, sequences):
        self.logger.info('Converting training data to StarSpace format\n')
        with open(self.interactions_fpath, "w") as out_file:
            for seq in tqdm(sequences):
                seq_line = " ".join(["{}_{}".format(self.label, s) for s in seq]) + "\n"
                out_file.write(seq_line)

    def fit(self, train_data):
        sequences = train_data['sequence'].values
        path = Path("tmp")
        path.mkdir(parents=True, exist_ok=True)
        self.convert_to_starspace_input(sequences)
        self.train_starspace()
        self.convert_to_wv_format()
        self.load_keyed_vectors()

    def exponential_decay_recommendation(self, user_profile, candidate_selector_f):
        rec = []
        try:
            # iterate the user profile backwards
            for i, item in enumerate(user_profile[::-1]):
                ms = candidate_selector_f(item)
                # apply exponential decay to the similarity scores
                decay = self.decay_alpha ** i
                ms = [(x[0], decay * x[1]) for x in ms]
                rec.extend(ms)
            # sort items by similarity score
            rec = sorted(rec, key=lambda x: -x[1])
        except KeyError:
            rec = []
        return [([x[0].replace(self.label + "_", "")], x[1]) for x in rec]

    def recommend(self, user_profile, user_id=None):
        f = lambda x: self.model.most_similar(positive=x)
        user_profile = ["{}_{}".format(self.label, up) for up in user_profile]
        return self.exponential_decay_recommendation(user_profile,
                                                     candidate_selector_f=f)


class StarSpaceContentRecommender(StarSpaceRecommender, WithMetadata):
    label = "DOC"
    model = None
    # keeps only the vectors starting with <label>_
    model_only_doc_tokens = None
    model_only_doc_tokens_path = "tmp/starspace_doc_only.model"
    # stores document vectors
    model_doc = None
    model_doc_path = "tmp/avg_doc_only.model"
    recommendation_selector = "default"

    # max tokens per entry
    max_tokens = 60

    def __init__(self, metadata_path, recommendation_selector="default",
                 max_tokens=60, **kwargs):
        """
        :param metadata_path: str: path to jsonl file containing metadata
        :param recommendation_selector: str: sets the recommendation algorithm. used when doing recommendations (not at training time). 
        this variable can take the following values:
       
         - "default": see self.default_recommend docs
        - "build_content_vector": see self.recommend_build_content docs 
         - "build_sequence_vector": see self.recommend_build_sequence docs
        """
        super(StarSpaceContentRecommender, self).__init__(**kwargs)
        self.max_tokens = max_tokens
        self.recommendation_selector = recommendation_selector
        self.logger.info("Loading metadata...")
        self.metadata = self.load_metadata(metadata_path)
        self.logger.info("Finished loading metadata...")

    def metadata_tokens(self, item_id: int) -> List[str]:
        """
        given a document id it returns a list of metadata tokens 
        which corresponds to the training representacion for a document item
        """
        tokens = []
        if item_id in self.metadata:
            properties = self.metadata[item_id]['properties']
            tokens = [":".join(p['property'], p['value']) for p in properties]
        return tokens

    def transform_session_into_starspace_entry(self, session: List[str]) -> str:
        """
        given a session (list of documents)
        it returns a line to add to starspace training corpus
        this line takes the following format:
        <token_doc_1> <token_doc_1><tab> <token_doc_2> <token_doc_2>...
        """
        entries = list()
        for s in session:
            if s is not None:
                item_token = "{}_{}".format(self.label, str(s))
                tokens = self.metadata_tokens(int(s))
                filtered_metadata_tokens = list()
                n_tokens = 0
                for token in tokens:
                    if "CATEGORY_" not in token:
                        if n_tokens < self.max_tokens:
                            n_tokens = n_tokens + 1
                            filtered_metadata_tokens.append(token)
                    else:
                        filtered_metadata_tokens.append(token)
                entries.append(" ".join([item_token] + filtered_metadata_tokens))
        return "\t".join(entries)

    def convert_to_starspace_input(self, sequences: List[List[str]]):
        """
        :param sequences: list of sessions i.e: [ [doc1, doc2..], [doc3, doc4].. ]
        given a list of sessions.
        it creates a starspace training corpus.
        """
        self.logger.info('Converting training data to StarSpace format\n')
        print("starting gneerating: %s" % self.interactions_fpath)
        with open(self.interactions_fpath, "w") as out_file:
            for seq in tqdm(sequences):
                l = self.transform_session_into_starspace_entry(seq)
                seq_line = l + "\n"
                out_file.write(seq_line)
        print("finished gneerating: %s" % self.interactions_fpath)

    def train_starspace(self):
        arg_list = [
             STARSPACE, "train",
             "-trainFile", self.interactions_fpath,
             "-model", self.model_fpath,
             "-label", self.label,
             "-loss", "softmax",  # default is hinge, but that is closer to word2vec
             "-trainMode", "1",
             "-fileFormat", "labelDoc",
             "-epoch", str(self.epoch),
             "-minCount", str(self.min_count),
             "-dim", str(self.size),
             "-thread", str(int(self.workers * 2)),
             "-maxNegSamples", "5"  # to make star space similar to word2vec
             ]
        self.run_cmd(arg_list)

    def convert_to_wv_format(self):
        self.logger.info("Converting to Word2Vec format.")
        txt_file_model_vectors = self.model_only_doc_tokens_path + ".txt"
        tsv_file = self.model_fpath + ".tsv"
        txt_file = self.model_fpath + ".txt"
        doc_vector_n = 0
        n_vec = self.file_len(tsv_file)
        header = "{} {}\n".format(n_vec, self.size)

        # dumps all vectors
        with open(txt_file, "w") as out_file:
            out_file.write(header)
            with open(tsv_file) as in_file:
                for l in tqdm(in_file):
                    if "{}_".format(self.label) in l:
                        doc_vector_n = doc_vector_n  + 1
                    out_file.write(l.replace("\t", " "))

        # dumps vectors only with <LABEL>_
        header = "{} {}\n".format(doc_vector_n, self.size)
        with open(txt_file_model_vectors, "w") as out:
            out.write(header)
            with open(tsv_file) as in_file:
                for l in tqdm(in_file):
                    if "{}_".format(self.label) in l:
                        out.write(l.replace("\t", " "))

    def default_recommend(self, user_profile, user_id=None):
        """
        :param user_profile: list of docs i.e: [doc1, doc2, doc3]
        During training we generated <LABEL>_<document> vectors. 
        we use those as document vectors. Iterate the user_profile backwards looking for similar documents at each time.
        """
        user_profile = ["{}_{}".format(self.label, up) for up in user_profile]
        f = lambda x: self.model_only_doc_tokens.most_similar(positive=x)
        return self.exponential_decay_recommendation(user_profile, candidate_selector_f=f)

    def get_doc_vector(self, itemid: int) -> List[float]:
        """
        :param itemid: identifier of a document
        :returns: a document vector

        given a document id, it generates a document vector using document metadata.
        
        - get all tokens for a document from metadata
        - get a vector for each token (generated during training)
        - average all token vectors
        
        this might not be the correct way to generate vectors for docs.
        worth taking a look at 'embed_doc' to generate document vectors
        it seems it is not just a simple vector average. [1]
        
        there seems to be a bug with 'embed_doc' in which it expects
        an uneeded argument.[2]

        there is also some good conv on this in [3]

        [1] https://github.com/facebookresearch/StarSpace/issues/282
        [2] https://github.com/facebookresearch/StarSpace/issues/293
        
        https://github.com/facebookresearch/StarSpace/issues/169
        """
        vectors = []
        if itemid in self.metadata.keys():
            document_tokens = self.metadata_tokens(itemid)
            for token in document_tokens:
                if token in self.model:
                    vector = self.model.get_vector(token).tolist()
                    vectors.append(vector)
        if vectors:
            return np.mean(vectors, axis=0).tolist()
        else:
            return []

    def get_seq_vector(self, seq: List[int]) -> List[float]:
        """
        :param seq: list of interactions iwth documents i.e: [doc1, doc2]
        :return sequence vector:
        builds the vector for a sequence of interactions.
        1. builds vectors for each document in the interactions
        2. average the vectors of all documents
        """
        vectors = []
        for s in seq:
            vector = self.get_doc_vector(s)
            if vector:
                vectors.append(vector)
        if vectors:
            return np.mean(vectors, axis=0).tolist()
        else:
            return []

    def recommend_build_content(self, user_profile, user_id=None):
        """
        :param user_profile: list of interactions iwth documents i.e: [doc1, doc2]
        Unlike "default", we consider a document vector to be the average of the vectors of the tokens in the document. 
        At recommendation time we Iterate the sequence backwards. in each step, we take a document vector and search for  similar documents to return as recommmendations.
        """
        def f(x):
            doc_vector = self.get_doc_vector(int(x))
            if not doc_vector:
                return []
            return self.model_doc.similar_by_vector(np.array(doc_vector), topn=10)
        recs = self.exponential_decay_recommendation(user_profile, candidate_selector_f=f)
        return recs
        
    def recommend_build_sequence(self, user_profile, user_id=None):
        """
        :param user_profile: list of interactions iwth documents i.e: [doc1, doc2]
 we consider a document vector to be the average of the vectors of the tokens in the document. When doing a recommendation, we build a single vector by averaging the document vectors  in the sequence. Recommendation would be the top documents with highest similarity
        do recommendation by building a single vector for the user_profile.
        averaging the vectors of all its documents.
        """
        user_profile = [int(i) for i in user_profile]
        seq_vector = self.get_seq_vector(user_profile)
        rec = []
        if seq_vector:
            rec = self.model_doc.similar_by_vector(np.array(seq_vector))
        return [([x[0]], x[1]) for x in rec]

    def recommend(self, user_profile, user_id=None):
        if self.recommendation_selector == "default":
            return self.default_recommend(user_profile=user_profile, user_id=user_id)
        if self.recommendation_selector == "build_content_vector":
            return self.recommend_build_content(user_profile=user_profile, user_id=user_id)
        if self.recommendation_selector == "build_sequence_vector":
            return self.recommend_build_sequence(user_profile=user_profile, user_id=user_id)

    def build_doc_model(self):
        """
        internal method.
        it creates a word2vec index with vectors each document existing in metadata. 
        this is called after starspace has run and we got vectors for each metadata token.
        each document vector is the average of its token vectors.
        """
        self.logger.info("generating doc vectors..")
        with open(self.model_doc_path, 'w') as o:
            keys = list(self.metadata.keys())
            n_vec = len(keys)
            header = "{} {}\n".format(n_vec, self.size)
            o.write(header)
            for itemid in tqdm(keys):
                vector = self.get_doc_vector(itemid)
                if vector:
                    str_vector = " ".join([str(i) for i in vector])
                    line = str(itemid) + " " + str_vector + "\n"
                    o.write(line)
        self.logger.info("loading doc vectors..")
        self.model_doc = KeyedVectors.load_word2vec_format(self.model_doc_path)

    def fit(self, train_data):
        # train like usual but then use metadata to build a vector per document
        super(StarSpaceContentRecommender, self).fit(train_data)
        self.build_doc_model()

    def load_keyed_vectors(self):
        path = self.model_only_doc_tokens_path + ".txt"
        self.model_only_doc_tokens = KeyedVectors.load_word2vec_format(path)
        super(StarSpaceContentRecommender, self).load_keyed_vectors()
