import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

import itertools

def ACC(n_classes, GT, pred):
    
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            for k in range(len(GT)):
                if GT[k] == i and pred[k] == j:
                    weights[i,j] += 1
    
    gt_ind, pr_ind = linear_sum_assignment(-weights)
    corrent_count = weights[gt_ind, pr_ind].sum()

    return corrent_count / len(GT)

def NMI(GT, pred):
    return normalized_mutual_info_score(GT, pred)

def ARI(GT, pred):
    return adjusted_rand_score(GT, pred)
