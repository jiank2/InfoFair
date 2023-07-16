import pickle

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from utils.utils import normalize_cols, encode_grouping, get_grouping


class Dataset:
    def __init__(self, dataset, device_name):
        self.name = dataset
        self.device = torch.device(device_name)

        # load features and labels
        self.features = np.load('../data/{dataset}/features.npy'.format(
            dataset=dataset
        ))
        self.features = torch.FloatTensor(
            normalize_cols(self.features)
        ).to(self.device)
        self.labels = torch.LongTensor(
            np.load("../data/{dataset}/labels.npy".format(
                dataset=dataset))
        ).to(self.device)

        # load indices
        self.train_idx = torch.LongTensor(
            np.load("../data/{dataset}/train_idx.npy".format(dataset=dataset))
        ).to(self.device)
        self.test_idx = torch.LongTensor(
            np.load("../data/{dataset}/test_idx.npy".format(dataset=dataset))
        ).to(self.device)
        ntest = int(self.test_idx.shape[0] / 2)
        self.val_idx = self.test_idx[:ntest]
        self.test_idx = self.test_idx[ntest:]
        del ntest

        # get stats
        self.num_train = len(self.train_idx)
        self.num_val = len(self.val_idx)
        self.num_test = len(self.test_idx)
        self.num_classes = self.labels.max().item() + 1
        self.possible_labels = list(range(self.num_classes))

    def create_sensitive_features(self, sensitive_attr):
        self.sensitive_attrs = sensitive_attr.strip().split(',')
        self.num_sensitive_attrs = len(self.sensitive_attrs)

        self.sensitive_grouping = encode_grouping(self.name, self.sensitive_attrs)

        self.sensitive_labels = torch.LongTensor(
            get_grouping(self.name, self.sensitive_attrs, self.sensitive_grouping)
        ).to(self.device)

        self.num_sensitive_groups = self.sensitive_labels.max().item() + 1

    def create_dataloader(self, train_batch_size):
        self.train_loader = DataLoader(
            TensorDataset(
                self.features[self.train_idx, :],
                self.labels[self.train_idx],
                self.sensitive_labels[self.train_idx]
            ),
            batch_size=train_batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(
                self.features[self.val_idx, :],
                self.labels[self.val_idx],
                self.sensitive_labels[self.val_idx]
            ),
            batch_size=len(self.val_idx),
            shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(
                self.features[self.test_idx, :],
                self.labels[self.test_idx],
                self.sensitive_labels[self.test_idx]
            ),
            batch_size=len(self.test_idx),
            shuffle=False
        )

    def remove_sensitive_features(self):
        with open('../data/{name}/headers_dict.pickle'.format(name=self.name), 'rb') as f:
            headers = pickle.load(f)

        indices = list()
        if isinstance(self.sensitive_attrs, str):
            indices += list(headers[self.sensitive_attrs].values())
        else:
            for attr in self.sensitive_attrs:
                indices += list(headers[attr].values())
        indices_to_keep = list(set(range(self.features.size()[1])) - set(indices))
        self.features = self.features[:, indices_to_keep]
        self.features = self.features.to(self.device)


class DisparateImpactDataset:
    def __init__(self, dataset, sensitive_attr, regularization, device_name):
        self.name = dataset
        self.sensitive_attr = sensitive_attr
        self.regularization = regularization
        self.device = torch.device(device_name)

        self.folder = "../data/disparate-impact/{dataset}/{regularization}".format(
            dataset=self.name,
            regularization=self.regularization
        )
        self.prefix = self.sensitive_attr if self.sensitive_attr in ("sex", "race", "marital") else "both"
        # load features and labels
        self.features = np.load("{folder}/{prefix}-repaired-features.npy".format(
            folder=self.folder,
            prefix=self.prefix
        ))
        self.features = torch.FloatTensor(
            normalize_cols(self.features)
        ).to(self.device)
        self.labels = torch.LongTensor(
            np.load("{folder}/{prefix}-repaired-labels.npy".format(
                folder=self.folder,
                prefix=self.prefix
            ))
        ).to(self.device)

        # load indices
        self.train_idx = torch.LongTensor(
            np.load("../data/{dataset}/train_idx.npy".format(dataset=dataset))
        ).to(self.device)
        self.test_idx = torch.LongTensor(
            np.load("../data/{dataset}/test_idx.npy".format(dataset=dataset))
        ).to(self.device)
        ntest = int(self.test_idx.shape[0] / 2)
        self.val_idx = self.test_idx[:ntest]
        self.test_idx = self.test_idx[ntest:]
        del ntest

        # get stats
        self.num_train = len(self.train_idx)
        self.num_val = len(self.val_idx)
        self.num_test = len(self.test_idx)
        self.num_classes = self.labels.max().item() + 1
        self.possible_labels = list(range(self.num_classes))

    def create_sensitive_features(self, sensitive_attr):
        self.sensitive_attrs = sensitive_attr.strip().split(',')
        self.num_sensitive_attrs = len(self.sensitive_attrs)

        self.sensitive_grouping = encode_grouping(self.name, self.sensitive_attrs)

        self.sensitive_labels = torch.LongTensor(
            get_grouping(self.name, self.sensitive_attrs, self.sensitive_grouping)
        ).to(self.device)

        self.num_sensitive_groups = self.sensitive_labels.max().item() + 1

    def create_dataloader(self, train_batch_size):
        self.train_loader = DataLoader(
            TensorDataset(
                self.features[self.train_idx, :],
                self.labels[self.train_idx],
                self.sensitive_labels[self.train_idx]
            ),
            batch_size=train_batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(
                self.features[self.val_idx, :],
                self.labels[self.val_idx],
                self.sensitive_labels[self.val_idx]
            ),
            batch_size=len(self.val_idx),
            shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(
                self.features[self.test_idx, :],
                self.labels[self.test_idx],
                self.sensitive_labels[self.test_idx]
            ),
            batch_size=len(self.test_idx),
            shuffle=False
        )

    def remove_sensitive_features(self):
        path = "{folder}/{prefix}-repaired-headers_dict.pickle".format(
            folder=self.folder,
            prefix=self.prefix
        )
        with open(path, "rb") as f:
            headers = pickle.load(f)

        indices = list()
        if isinstance(self.sensitive_attrs, str):
            indices += list(headers[self.sensitive_attrs].values())
        else:
            for attr in self.sensitive_attrs:
                indices += list(headers[attr].values())
        indices_to_keep = list(set(range(self.features.size()[1])) - set(indices))
        self.features = self.features[:, indices_to_keep]
        self.features = self.features.to(self.device)
