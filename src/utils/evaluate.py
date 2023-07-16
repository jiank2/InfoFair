import numpy as np
import sklearn
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def accuracy(ytrue, ypred):
    try:
        ytrue, ypred = torch.flatten(ytrue), torch.flatten(ypred)
        ytrue, ypred = ytrue.cpu().detach().numpy(), ypred.cpu().detach().numpy()
    except:
        ytrue, ypred = ytrue.flatten(), ypred.flatten()
    acc = np.sum(ytrue == ypred) / ytrue.shape[0]
    return acc


def f1(ytrue, ypred, type="binary"):
    try:
        ytrue, ypred = torch.flatten(ytrue), torch.flatten(ypred)
        ytrue, ypred = ytrue.cpu().detach().numpy(), ypred.cpu().detach().numpy()
    except:
        ytrue, ypred = ytrue.flatten(), ypred.flatten()
    res = f1_score(ytrue, ypred, average=type)
    return res


def precision(ytrue, ypred, type="binary"):
    try:
        ytrue, ypred = torch.flatten(ytrue), torch.flatten(ypred)
        ytrue, ypred = ytrue.cpu().detach().numpy(), ypred.cpu().detach().numpy()
    except:
        ytrue, ypred = ytrue.flatten(), ypred.flatten()
    res = precision_score(ytrue, ypred, average=type, zero_division=0)
    return res


def recall(ytrue, ypred, type="binary"):
    try:
        ytrue, ypred = torch.flatten(ytrue), torch.flatten(ypred)
        ytrue, ypred = ytrue.cpu().detach().numpy(), ypred.cpu().detach().numpy()
    except:
        ytrue, ypred = ytrue.flatten(), ypred.flatten()
    res = recall_score(ytrue, ypred, average=type, zero_division=0)
    return res


def get_statistical_parity_stats(ypred, members, nclass, nsensitive):
    res = dict()
    for i in range(nsensitive):
        res[i] = dict()
        for j in range(nclass):
            res[i][j] = 0
    try:
        ypred, members = torch.flatten(ypred), torch.flatten(members)
        ypred, members = ypred.cpu().detach().numpy(), members.cpu().detach().numpy()
    except:
        ypred, members = ypred.flatten(), members.flatten()
    nsample = ypred.shape[0]
    for i in range(nsample):
        label, membership = ypred[i], members[i]
        res[round(membership)][round(label)] += 1
    return res


def statistical_imparity(
    stats, possible_labels, single_reduction="mean", total_reduction="mean"
):
    """
    For any a1 and a2 (a1 != a2),
    if single_reduction == 'mean':
        bias(y) = avg(|P(Y = y|a = a1) - P(Y = y|a = a2)|)
    elif single_reduction == 'max':
        bias(y) = max(|P(Y = y|a = a1) - P(Y = y|a = a2)|)

    For any y,
    if total_reduction == 'mean':
        total_bias = avg(bias(y)) for all y
    elif total_reduction == 'max':
        total_bias = max(bias(y)) for all y
    """
    single_imparity = [[] for _ in possible_labels]
    nmember = len(stats)
    members = list(stats.keys())

    for i in range(nmember):
        for j in range(nmember):
            if i <= j:
                m1, m2 = members[i], members[j]
                for y in possible_labels:
                    if sum(list(stats[m1].values())) == 0:
                        p_y_m1 = 0
                    else:
                        p_y_m1 = stats[m1][y] / sum(list(stats[m1].values()))
                    if sum(list(stats[m2].values())) == 0:
                        p_y_m2 = 0
                    else:
                        p_y_m2 = stats[m2][y] / sum(list(stats[m2].values()))
                    diff = abs(p_y_m1 - p_y_m2)
                    single_imparity[y].append(diff)

    if single_reduction == "mean":
        single_imparity = [np.mean(x) for x in single_imparity]
    elif single_reduction == "max":
        single_imparity = [np.max(x) for x in single_imparity]
    else:
        print("WARNING: single reduction should be either mean or max!")

    if total_reduction == "mean":
        total_imparity = np.mean(single_imparity)
    elif total_reduction == "max":
        total_imparity = np.max(single_imparity)
    else:
        total_imparity = -1
        print("WARNING: total reduction should be either mean or max!")
    return total_imparity


"""
helper functions
"""
def average(lst):
    return sum(lst) / len(lst)
