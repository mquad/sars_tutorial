from recommenders.FreqSeqMining import FreqSeqMiningRecommender
import pandas as pd
from util.split import random_holdout
from util import evaluation

##laod data
import ast
db=pd.read_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',converters={'songs':ast.literal_eval})
seqs = db['songs'].tolist()

train_seq,test_seq = random_holdout(seqs, 0.8)

rec = FreqSeqMiningRecommender(0.005,0.1,False)
rec.fit(train_seq)

max_context=10
min_context=1
recommendation_length=1

evaluation.set_evaluation(rec,test_seq,max_context,min_context,recommendation_length,[evaluation.precision,evaluation.recall])


def sequential_evaluation(recommender,test_seq,max_context,min_context,recommendation_length,evaluation_functions):

    if max_context == len(test_seq) - 1:
        return evaluate_sequence(recommender,test_seq,max_context,min_context,recommendation_length,evaluation_functions)
    elif max_context < len(test_seq) - 1:
        return (evaluate_sequence(recommender,test_seq,max_context,min_context,recommendation_length,evaluation_functions) + sequential_evaluation(recommender,test_seq,max_context+1,min_context,recommendation_length,evaluation_functions)) / (len(test_seq)-max_context)
    else:
        raise NameError('Trying to evaluate without ground truth')


def evaluate_sequence(recommender,test_seq,max_context,min_context,recommendation_length,evaluation_functions):#TODO add look-ahead
    r = recommender.recommend(test_seq, max_context=max_context, min_context=min_context, recommendation_length=recommendation_length)
    ground_truth = [test_seq[max_context:max_context+1]] #TODO if recomendation length >1
    tmpResults = []
    for f in evaluation_functions:
        tmpResults.append(f(ground_truth,recommender.get_recommendation_list(r)))
    return np.array(tmpResults)