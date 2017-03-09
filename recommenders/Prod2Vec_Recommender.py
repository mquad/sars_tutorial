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


    def __init__(self,min_count=10,size=100,workers=4,window=5):
        '''
        Define Prod2Vec recommender
        :param min_count: if a word appears less than min_count times it is pruned
        :param size: number of layers in NN
        :param workers: parallelization
        :param window  is the maximum distance between the current and predicted word within a sentence.
        :param topn Number of recommendations
        :return: Nothing
        '''
        super(Prod2VecRecommender, self).__init__()
        self.workers = workers
        self.min_count=min_count
        self.size =size
        self.window = window

    def fit(self, sequences):
        self.model = gensim.models.Word2Vec(sequences, min_count=self.min_count,window=self.window,hs=1,size=self.size,sg=1,workers=self.workers)

    def recommend(self, user_profile):
        try:
            rec = self.model.most_similar(positive=user_profile)
        except KeyError:
            rec=[]
        return [([x[0]],x[1]) for x in rec] #use format as other recommenders

