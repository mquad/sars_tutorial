import numpy as np
from tqdm import tqdm

def sequential_evaluation(recommender, test_sequences, last_k, look_ahead, evaluation_functions,top_n):

    if last_k==0:
        raise NameError('0 doesn"t make sense as last_k')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for s in test_sequences:
            metrics += sequence_sequential_evaluation(recommender, s, last_k, evaluation_functions,top_n)
            pbar.update(1)
    return metrics/len(test_sequences)

def sequential_evaluation_use_user(recommender, test_sequences, users,last_k, look_ahead, evaluation_functions,top_n):

    if last_k==0:
        raise NameError('0 doesn"t make sense as last_k')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for i,s in enumerate(test_sequences):
            metrics += sequence_sequential_evaluation_use_user(recommender, s,users[i], last_k, evaluation_functions,top_n)
            pbar.update(1)
    return metrics/len(test_sequences)


def set_evaluation(recommender, test_sequences, last_k, look_ahead, evaluation_functions,top_n):
    '''
    :param recommender: recommender to use
    :param test_sequences: list of test sequences
    :param last_k: last_k element to consider as ground truth for each test sequence.
    If<0 it is interpreted as the first k elements for the user_profile
    :param look_ahead:number of elements in the look ahead
    :param evaluation_functions: metrics to use to evaluate performances
    :return: array of resulting metrics
    '''

    if last_k==0:
        raise NameError('0 doesn"t make sense as last_k')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for s in test_sequences:
            metrics += evaluate_sequence(recommender, s, last_k, look_ahead, evaluation_functions,top_n)
            pbar.update(1)
    return metrics/len(test_sequences)

def set_evaluation_use_user(recommender, test_sequences, users, last_k, look_ahead, evaluation_functions,top_n):
    '''
    :param recommender: recommender to use
    :param test_sequences: list of test sequences
    :param last_k: last_k element to consider as ground truth for each test sequence.
    If<0 it is interpreted as the first k elements for the user_profile
    :param look_ahead:number of elements in the look ahead
    :param users: users corresponding to the test sequences
    :param evaluation_functions: metrics to use to evaluate performances
    :return: array of resulting metrics
    '''

    if last_k==0:
        raise NameError('0 doesn"t make sense as last_k')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for i,s in enumerate(test_sequences):
            metrics += evaluate_sequence_use_user(recommender, s,users[i], last_k, look_ahead, evaluation_functions,top_n)
            pbar.update(1)
    return metrics/len(test_sequences)

def evaluate_sequence(recommender, seq, last_k, look_ahead, evaluation_functions,top_n):
    """
    :param recommender: which recommender to use
    :param seq: the user_profile/ context
    :param last_k: last element used as ground truth. NB if <0 it is interpreted as first elements to keep
    :param evaluation_functions: which function to use to evaluate the rec performance
    :param look_ahead: number of elements in ground truth to consider. if look_ahead = 'all' then all the ground_truth sequence is considered
    :return: performance of recomender
    """

    ##if last_k <0 actually it means first_m
    if last_k <= 0:
        last_k = len(seq)+last_k

    user_profile = seq[:-last_k]
    ground_truth = seq[-last_k:]
    #restrict ground truth to look_ahead
    ground_truth = ground_truth[:look_ahead]  if look_ahead != 'total' else ground_truth
    ground_truth = list(map(lambda x:[x],ground_truth)) #list of list format

    if not user_profile or not ground_truth:
        # if any of the two missing
        # all evaluation functions are 0
        return np.zeros(len(evaluation_functions))

    r = recommender.recommend(user_profile)[:top_n]

    if not r:
        # no recommendation found
        return np.zeros(len(evaluation_functions))

    tmpResults = []
    for f in evaluation_functions:
        tmpResults.append(f(ground_truth,recommender.get_recommendation_list(r)))
    return np.array(tmpResults)


def evaluate_sequence_use_user(recommender, seq, user, last_k, look_ahead, evaluation_functions,top_n):

    ##safety checks
    if last_k <= 0:
        last_k = len(seq)+last_k

    user_profile = seq[:-last_k]
    ground_truth = seq[-last_k:]
    #restrict ground truth to look_ahead
    ground_truth = ground_truth[:look_ahead]  if look_ahead != 'total' else ground_truth
    ground_truth = list(map(lambda x:[x],ground_truth)) #list of list format

    if not user_profile or not ground_truth:
        # if any of the two missing
        # all evaluation functions are 0
        return np.zeros(len(evaluation_functions))

    r = recommender.recommend(user_profile,user)[:top_n]

    if not r:
        # no recommendation found
        return np.zeros(len(evaluation_functions))

    tmpResults = []
    for f in evaluation_functions:
        tmpResults.append(f(ground_truth,recommender.get_recommendation_list(r)))
    return np.array(tmpResults)



def _sse(recommender, seq, last_k, evaluation_functions,top_n):

    if last_k == 1:
        return evaluate_sequence(recommender, seq, last_k, 1, evaluation_functions,top_n)
    else:
        return (evaluate_sequence(recommender, seq, last_k, 1, evaluation_functions,top_n) + \
                _sse(recommender, seq, last_k - 1, evaluation_functions,top_n))

def _sse_use_user(recommender, seq, user, last_k, evaluation_functions,top_n):

    if last_k == 1:
        return evaluate_sequence_use_user(recommender, seq, user, last_k, 1, evaluation_functions,top_n)
    else:
        return (evaluate_sequence_use_user(recommender, seq, user, last_k, 1, evaluation_functions,top_n) + \
                _sse_use_user(recommender, seq, user, last_k - 1, evaluation_functions,top_n))

def sequence_sequential_evaluation(recommender, seq, last_k, evaluation_functions,top_n):
    if last_k <= 0:
        last_k = len(seq)+last_k
    return _sse(recommender, seq, last_k, evaluation_functions,top_n) / last_k

def sequence_sequential_evaluation_use_user(recommender, seq, user, last_k, evaluation_functions,top_n):
    if last_k <= 0:
        last_k = len(seq)+last_k
    return _sse_use_user(recommender, seq, user, last_k, evaluation_functions,top_n) / last_k