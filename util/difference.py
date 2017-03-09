import numpy as np

precision_popularity =np.matrix([[0.0218,0.0329,0.045],
                                [0.0058,0.0069,0.0072]])
precision_fpmc = np.matrix([[0.2816,0.2304,0.1668],
                            [0.1527,0.0674,0.0360]])
precision_fpm=np.matrix([[0.1878,0.2592,0.2128],
                         [0.1112,0.1284,0.0982]])
precision_markov=np.matrix([[0.0237,0.0897,0.1412],
                            [0.0084,0.0314,0.0489]])
precision_prod2vec=np.matrix([[0.2402,0.2120,0.1153],
                              [0.1304,0.0614,0.0260]])
precision_supervised=np.matrix([[0.0134,0.0591,0.0572],
                                [0.0072,0.0192,0.0120]])



recs = [precision_fpmc,precision_fpm,precision_markov,precision_popularity,precision_prod2vec,precision_supervised]

diff = np.zeros((1,precision_fpm.shape[1]))
for r in recs:
    for r2 in recs:
        #print(abs(r[0]-r2[0]) - abs(r[1]-r2[1]))
        diff += abs(r[0]-r2[0]) - abs(r[1]-r2[1])

diff = diff/2

recall_popularity =np.matrix([[0.0058,0.0349,0.0727],
                              [0.0058,0.0349,0.0726]])
recall_fpmc = np.matrix([[0.1374,0.3290,0.3534],
                         [0.1527,0.3371,0.3598]])
recall_fpm=np.matrix([[0.1958,0.3570,0.2529],
                      [0.2198,0.3968,0.3011]])
recall_markov=np.matrix([[0.0393,0.1262,0.1906],
                         [0.0434,0.1570,0.2448]])
recall_prod2vec=np.matrix([[0.1015,0.2869,0.2474],
                           [0.1304,0.3074,0.2601]])
recall_supervised=np.matrix([[0.0215,0.0662,0.0532],
                             [0.0363,0.0963,0.0604]])



recs = [recall_fpmc,recall_fpm,recall_markov,recall_popularity,recall_prod2vec,recall_supervised]

diff_recall = np.zeros((1,recall_fpm.shape[1]))
for r in recs:
    for r2 in recs:
       # print(abs(r[0]-r2[0]) - abs(r[1]-r2[1]))
        diff_recall += abs(r[0]-r2[0]) - abs(r[1]-r2[1])

diff_recall = diff_recall/2
