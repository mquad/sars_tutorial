import numpy as np
from tqdm import tqdm

def sequential_evaluation(recommender, test_sequences, last_k, look_ahead, evaluation_functions):
    if last_k <= 0:
        raise NameError('Please choose a k>0')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for s in test_sequences:
            metrics += sequence_sequential_evaluation(recommender, s, last_k, evaluation_functions)
            pbar.update(1)
    return metrics/len(test_sequences)


def set_evaluation(recommender, test_sequences, last_k, look_ahead, evaluation_functions):
    if last_k <= 0:
        raise NameError('Please choose a k>0')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for s in test_sequences:
            metrics += evaluate_sequence(recommender, s, last_k, look_ahead, evaluation_functions)
            pbar.update(1)
    return metrics/len(test_sequences)

def evaluate_sequence(recommender, seq, last_k, look_ahead, evaluation_functions):
    """
    :param recommender:
    :param seq:
    :param evaluation_functions:
    :param look_ahead: number of elements in ground truth to consider. if look_ahead = 'all' then all the ground_truth sequence is considered
    :return:
    """

    ##safety checks
    if last_k <= 0:
        raise NameError('Please choose a k>0')

    user_profile = seq[:-last_k]
    ground_truth = seq[-last_k:]
    #restrict ground truth to look_ahead
    ground_truth = ground_truth[:look_ahead]  if look_ahead != 'total' else ground_truth
    ground_truth = list(map(lambda x:[x],ground_truth)) #list of list format

    if not user_profile or not ground_truth:
        # if any of the two missing
        # all evaluation functions are 0
        return np.zeros(len(evaluation_functions))

    r = recommender.recommend(user_profile)

    #print('user profile',user_profile)
    #print('ground_truth',ground_truth)
    #print('rec',recommender.get_recommendation_list(r))

    if not r:
        # no recommendation found
        return np.zeros(len(evaluation_functions))

    tmpResults = []
    for f in evaluation_functions:
        tmpResults.append(f(ground_truth,recommender.get_recommendation_list(r)))
    return np.array(tmpResults)

def _sse(recommender, seq, last_k, evaluation_functions):

    if last_k <= 0:
        raise NameError('Please choose a k>0')
    elif last_k == 1:
        return evaluate_sequence(recommender, seq, last_k, 1, evaluation_functions)
    else:
        return (evaluate_sequence(recommender, seq, last_k, 1, evaluation_functions) + \
                _sse(recommender, seq, last_k - 1, evaluation_functions))

def sequence_sequential_evaluation(recommender, seq, last_k, evaluation_functions):
    return _sse(recommender, seq, last_k, evaluation_functions) / last_k