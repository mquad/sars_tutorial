import numpy as np

def precision(ground_truth, prediction):
    """both lists of lists
    :param groud_truth:
    :param prediction:
    :return:
    """
    precision_score = count_a_in_b(prediction,ground_truth) / float(len(ground_truth))
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ground_truth, prediction):
    recall_score = 0 if len(prediction)==0 else count_a_in_b(prediction, ground_truth) / float(len(prediction))
    assert 0 <= recall_score <= 1
    return recall_score

def count_a_in_b(a,b):
    """
    :param a: list of lists
    :param b: list of lists
    :return: number of elements of a in b
    """
    count=0
    for el in a:
        if el in b:
            count +=1
    return count


def set_evaluation(recommender,test_seq,max_context,min_context,recommendation_length,evaluation_functions):
    count = 0
    results = np.zeros(len(evaluation_functions))

    for s in test_seq:
        #print(s)
        if max_context >= len(s): continue #non c'e' ground truth per controllare

        r = recommender.recommend(s, max_context=max_context, min_context=min_context, recommendation_length=recommendation_length)

        count += 1
        ground_truth = list(map(lambda x:[x],s[max_context:]))
        tmpResults = []
        for f in evaluation_functions:
            tmpResults.append(f(ground_truth,recommender.get_recommendation_list(r)))
        results += np.array(tmpResults)

    return results / (count if count != 0 else 1)
