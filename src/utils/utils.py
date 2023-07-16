import itertools
import pickle

import numpy as np
import torch


def normalize_cols(mat):
    """
    normalize each column by its largest absolute values

    :param mat: numpy ndarray

    :return: normalized matrix
    """
    abs_max_vals = np.abs(mat).max(axis=0)
    abs_max_vals = np.where(abs_max_vals == 0, 1, abs_max_vals)  # avoid divide by zero
    mat = mat / abs_max_vals
    return mat


def encode_grouping(dataset, attrs, is_baseline=False):
    if is_baseline:
        with open(
            "../../data/{dataset}/headers_dict.pickle".format(dataset=dataset), "rb"
        ) as f:
            headers = pickle.load(f)
    else:
        with open(
            "../data/{dataset}/headers_dict.pickle".format(dataset=dataset), "rb"
        ) as f:
            headers = pickle.load(f)
    groupings = list()
    for attr in attrs:
        nclass = len(headers[attr])
        if nclass == 1:
            groupings.append(["0", "1"])
        else:
            groupings.append(
                encode_onehot(list(range(nclass)), return_ndarray=False, to_string=True)
            )
    first = groupings[0]
    for i in range(1, len(groupings)):
        second = groupings[i]
        first = list(itertools.product(first, second))  # cartesian product
        first = ["".join(x) for x in first]
    res = {k: v for v, k in enumerate(first)}
    return res


def get_grouping(dataset, attrs, grouping, is_baseline=False):
    if is_baseline:
        with open(
            "../../data/{dataset}/headers_dict.pickle".format(dataset=dataset), "rb"
        ) as f:
            headers = pickle.load(f)
        features = np.load("../../data/{dataset}/features.npy".format(dataset=dataset))
    else:
        with open(
            "../data/{dataset}/headers_dict.pickle".format(dataset=dataset), "rb"
        ) as f:
            headers = pickle.load(f)
        features = np.load("../data/{dataset}/features.npy".format(dataset=dataset))

    indices = list()
    if isinstance(attrs, str):
        indices += list(headers[attrs].values())
    else:
        for attr in attrs:
            indices += list(headers[attr].values())
    sensitive_features = features[:, indices].astype(np.int16).tolist()
    sensitive_labels = np.array(
        [grouping["".join(map(str, individual))] for individual in sensitive_features]
    )
    return sensitive_labels


def encode_onehot(labels, return_ndarray=True, to_string=False):
    classes = set(labels)
    if return_ndarray:
        classes_dict = {
            c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
        }
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    else:
        classes_dict = {
            c: np.identity(len(classes), dtype=np.int32)[i, :].tolist()
            for i, c in enumerate(classes)
        }
        labels_onehot = list(map(classes_dict.get, labels))
        if to_string:
            labels_onehot = ["".join(map(str, x)) for x in labels_onehot]
    return labels_onehot


def label_to_one_hot(labels, nclass):
    """
    transform numeric label to one-hot encoding

    :param labels: a list of numeric labels, e.g., [0, 1, 1, 0, 0, 1] for binary classification
    :param nclass: numebr of classes

    :return: one-hot encodings of class labels
    """
    res = list()
    for label in labels:
        x = [0] * nclass
        x[label] = 1
        res.append(x)
    return res


def accuracy(prob, labels):
    """
    calculate accuracy of prediction

    :param prob: class probability
    :param labels: ground truth labels

    :return: accuracy
    """
    preds = prob.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    scipy sparse matrix to torch sparse tensor

    :param sparse_mx: scipy sparse matrix

    :return: torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    """
    adapted from PyTorch source code in case it is deprecated

    :param logits: log probability
    :param tau: temperature, should be nonnegative
    :param hard: whether to return hard one-hot vector
    :param dim: dimension along which softmax will be computed

    :return: sampled tensor with same shape as logits in from gumbel-softmax distribution
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
