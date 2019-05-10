from recommenders.ISeqRecommender import ISeqRecommender
from util.data_utils import dataset_to_gru4rec_format
from util.knn.iknn import ItemKNN
from util.knn.sknn import SessionKNN
from util.knn.vmsknn import VMSessionKNN
from util.knn.ssknn import SeqSessionKNN
from util.knn.sfsknn import SeqFilterSessionKNN


class KNNRecommender(ISeqRecommender):
    """
    Interface to ItemKNN and Session-based KNN methods. Based on:

    Evaluation of Session-based Recommendation Algorithms, Malte Ludewig and Dietmar Jannach
    """
    knn_models = {
        'iknn': ItemKNN,
        'sknn': SessionKNN,
        'v-sknn': VMSessionKNN,
        's-sknn': SeqSessionKNN,
        'sf-sknn': SeqFilterSessionKNN
    }

    def __init__(self,
                 model='cknn',
                 **init_args):
        """
        :param model: One among the following KNN models:
            - iknn: ItemKNN, item-to-item KNN based on the *last* item in the session to determine the items to be recommended.
            - sknn: SessionKNN, compares the *entire* current session with the past sessions in the training data to
                    determine the items to be recommended.
            - v-sknn: VMSessionKNN, use linearly decayed real-valued vectors to encode the current session,
                    then compares the current session with the past sessions in the training data using the dot-product
                    to determine the items to be recommended.
            - s-sknn: SeqSessionKNN, this variant also puts more weight on elements that appear later in the session by
                using a custom scoring function (see the paper by Ludewng and Jannach).
            - sf-sknn: SeqFilterSessionKNN, this variant also puts more weight on elements that appear later in the session
                in a more restrictive way by using a custom scoring function (see the paper by Ludewng and Jannach).

        :param init_args: The model initialization arguments. See the following initializations or
            check `util.knn` for more details on each model:
            - iknn: ItemKNN(n_sims=100, lmbd=20, alpha=0.5)
            - sknn: SessionKNN(k, sample_size=500, sampling='recent', similarity='jaccard', remind=False, pop_boost=0)
            - v-sknn: VMSessionKNN(k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div',
                 dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score',
                 weighting_time=False, normalize=True)
            - s-knn: SeqSessionKNN(k, sample_size=1000, sampling='recent', similarity='jaccard', weighting='div',
                remind=False, pop_boost=0, extend=False, normalize=True)
            - sf-sknn: SeqFilterSessionKNN(k, sample_size=1000, sampling='recent', similarity='jaccard', remind=False, pop_boost=0,
                 extend=False, normalize=True)
        """
        super(KNNRecommender).__init__()
        if model not in self.knn_models:
            raise ValueError("Unknown KNN model '{}'. The available ones are: {}".format(
                model, list(self.knn_models.keys())
            ))
        self.init_args = init_args
        self.init_args.update(dict(session_key='session_id',
                                   item_key='item_id',
                                   time_key='ts'))
        self.model = self.knn_models[model](**self.init_args)
        self.pseudo_session_id = 0

    def __str__(self):
        return str(self.model)

    def fit(self, train_data):
        self.logger.info('Converting training data to GRU4Rec format')
        # parse training data to GRU4Rec format
        train_data = dataset_to_gru4rec_format(dataset=train_data)

        self.logger.info('Training started')
        self.model.fit(train_data)
        self.logger.info('Training completed')
        self.pseudo_session_id = 0

    def recommend(self, user_profile, user_id=None):
        for item in user_profile:
            pred = self.model.predict_next(session_id=self.pseudo_session_id,
                                           input_item_id=item)
        # sort items by predicted score
        pred.sort_values(0, ascending=False, inplace=True)
        # increase the psuedo-session id so that future call to recommend() won't be connected
        self.pseudo_session_id += 1
        # convert to the required output format
        return [([x.index], x._2) for x in pred.reset_index().itertuples()]
