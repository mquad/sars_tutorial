import numpy as np

def sequential_evaluation(recommender, test_sequences, last_k, look_ahead, evaluation_functions):
    if last_k <= 0:
        raise NameError('Please choose a k>0')

    metrics = np.zeros(len(evaluation_functions))
    for s in test_sequences:
        metrics += sequence_sequential_evaluation(recommender, s, last_k, evaluation_functions)
    return metrics/len(test_sequences)


def set_evaluation(recommender, test_sequences, last_k, look_ahead, evaluation_functions):
    if last_k <= 0:
        raise NameError('Please choose a k>0')

    metrics = np.zeros(len(evaluation_functions))
    for s in test_sequences:
        metrics += evaluate_sequence(recommender, s, last_k, look_ahead, evaluation_functions)
    return metrics/len(test_sequences)

def evaluate_sequence(recommender, seq, last_k, look_ahead, evaluation_functions):
    """
    :param recommender:
    :param seq:
    :param last_k: if isLastK true the number of elements taken from the end of the list for the groud truth, is isLastK=False
            the number of elements taken from the beginnig of the seqeunce as the user_profile
    :param max_context:
    :param min_context:
    :param recommendation_length:
    :param evaluation_functions:
    :param is_last_k: if evaluation procedure lask-k, if false evaluation procedure is first-k.
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