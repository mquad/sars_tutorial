import numpy as np
from tqdm import tqdm


def sequential_evaluation(recommender,
                          test_sequences,
                          evaluation_functions,
                          users=None,
                          given_k=1,
                          look_ahead=1,
                          top_n=10,
                          scroll=True,
                          step=1):
    """
    Runs sequential evaluation of a recommender over a set of test sequences
    :param recommender: the instance of the recommender to test
    :param test_sequences: the set of test sequences
    :param evaluation_functions: list of evaluation metric functions
    :param users: (optional) the list of user ids associated to each test sequence. Required by personalized models like FPMC.
    :param given_k: (optional) the initial size of each user profile, starting from the first interaction in the sequence.
                    If <0, start counting from the end of the sequence. It must be != 0.
    :param look_ahead: (optional) number of subsequent interactions in the sequence to be considered as ground truth.
                    It can be any positive number or 'all' to extend the ground truth until the end of the sequence.
    :param top_n: (optional) size of the recommendation list
    :param scroll: (optional) whether to scroll the ground truth until the end of the sequence.
                If True, expand the user profile and move the ground truth forward of `step` interactions. Recompute and evaluate recommendations every time.
                If False, evaluate recommendations once per sequence without expanding the user profile.
    :param step: (optional) number of interactions that will be added to the user profile at each step of the sequential evaluation.
    :return: the list of the average values for each evaluation metric
    """
    if given_k == 0:
        raise ValueError('given_k must be != 0')

    metrics = np.zeros(len(evaluation_functions))
    with tqdm(total=len(test_sequences)) as pbar:
        for i, test_seq in enumerate(test_sequences):
            if users is not None:
                user = users[i]
            else:
                user = None
            if scroll:
                metrics += sequence_sequential_evaluation(recommender,
                                                          test_seq,
                                                          evaluation_functions,
                                                          user,
                                                          given_k,
                                                          look_ahead,
                                                          top_n,
                                                          step)
            else:
                metrics += evaluate_sequence(recommender,
                                             test_seq,
                                             evaluation_functions,
                                             user,
                                             given_k,
                                             look_ahead,
                                             top_n)
            pbar.update(1)
    return metrics / len(test_sequences)


def evaluate_sequence(recommender, seq, evaluation_functions, user, given_k, look_ahead, top_n):
    """
    :param recommender: which recommender to use
    :param seq: the user_profile/ context
    :param given_k: last element used as ground truth. NB if <0 it is interpreted as first elements to keep
    :param evaluation_functions: which function to use to evaluate the rec performance
    :param look_ahead: number of elements in ground truth to consider. if look_ahead = 'all' then all the ground_truth sequence is considered
    :return: performance of recommender
    """
    # safety checks
    if given_k < 0:
        given_k = len(seq) + given_k

    user_profile = seq[:given_k]
    ground_truth = seq[given_k:]

    # restrict ground truth to look_ahead
    ground_truth = ground_truth[:look_ahead] if look_ahead != 'all' else ground_truth
    ground_truth = list(map(lambda x: [x], ground_truth))  # list of list format

    if not user_profile or not ground_truth:
        # if any of the two missing all evaluation functions are 0
        return np.zeros(len(evaluation_functions))

    r = recommender.recommend(user_profile, user)[:top_n]

    if not r:
        # no recommendation found
        return np.zeros(len(evaluation_functions))
    reco_list = recommender.get_recommendation_list(r)

    tmp_results = []
    for f in evaluation_functions:
        tmp_results.append(f(ground_truth, reco_list))
    return np.array(tmp_results)


def sequence_sequential_evaluation(recommender, seq, evaluation_functions, user, given_k, look_ahead, top_n, step):
    if given_k < 0:
        given_k = len(seq) + given_k

    eval_res = None
    eval_cnt = 0
    for gk in range(given_k, len(seq), step):
        if eval_res is None:
            eval_res = evaluate_sequence(recommender, seq, evaluation_functions, user, given_k, look_ahead, top_n)
        else:
            eval_res += evaluate_sequence(recommender, seq, evaluation_functions, user, given_k, look_ahead, top_n)
        eval_cnt += 1
    return eval_res / eval_cnt
