import logging
from recommenders.ISeqRecommender import ISeqRecommender
import gensim

class Prod2VecRecommender(ISeqRecommender):
    """
    Creates Prod2Vec skipgram model from Grbovic, Mihajlo, Vladan Radosavljevic, Nemanja Djuric, Narayan Bhamidipati,
    Jaikit Savla, Varun Bhagwan, and Doug Sharp. "E-commerce in your inbox: Product recommendations at scale."
     In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
      pp. 1809-1818. ACM, 2015.
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def __init__(self):
        super(Prod2VecRecommender, self).__init__()
