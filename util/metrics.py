
def precision(ground_truth, prediction):
    """both lists of lists
    :param groud_truth:
    :param prediction:
    :return:
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    precision_score = count_a_in_b_unique(prediction, ground_truth) / float(len(prediction))
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ground_truth, prediction):
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    recall_score = 0 if len(prediction)==0 else count_a_in_b_unique(prediction, ground_truth) / float(len(ground_truth))
    assert 0 <= recall_score <= 1
    return recall_score

def count_a_in_b_unique(a, b):
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

def remove_duplicates(l):
    return [list(x) for x in set(tuple(x) for x in l)]
