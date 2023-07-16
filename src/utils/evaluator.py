import numpy as np
import torch
from sklearn.metrics import f1_score


class Evaluator:
    """
    a wrapper to evaluate the proposed method
    """

    def __init__(self, fairness, logger=None):
        self.fairness = fairness
        self.logger = logger

        self.result = self._default_result()

    def eval(
        self,
        loss,
        ypred,
        ytrue,
        sensitive_group_memberships,
        num_classes,
        num_sensitive_groups,
        stage,
    ):
        """
        evaluate the learned results

        :param loss: loss values
        :param ypred: predicted label
        :param ytrue: ground-truth label
        :param sensitive_group_memberships: a list of sensitive group indicator for each data point
        :param num_classes: number of class
        :param num_sensitive_groups: number of demographic groups
        :param stage: train, validation or test

        :return: a dict that saves all evaluation results
        """
        try:
            # flatten
            ypred = torch.flatten(ypred)
            sensitive_group_memberships = torch.flatten(sensitive_group_memberships)
            ytrue = torch.flatten(ytrue)
            # change to numpy array
            ypred = ypred.cpu().detach().numpy()
            sensitive_group_memberships = (
                sensitive_group_memberships.cpu().detach().numpy()
            )
            ytrue = ytrue.cpu().detach().numpy()
        except:
            ypred = ypred.flatten()
            sensitive_group_memberships = sensitive_group_memberships.flatten()
            ytrue = ytrue.flatten()

        # calculate evaluation metrics
        self.result["stage"] = stage
        self.result["loss"] = loss
        self.result["accuracy"] = self._calc_accuracy(ytrue=ytrue, ypred=ypred)
        self.result["binary_f1"] = self._calc_f1(
            ytrue=ytrue, ypred=ypred, type="binary"
        )
        self.result["micro_f1"] = self._calc_f1(ytrue=ytrue, ypred=ypred, type="micro")
        self.result["macro_f1"] = self._calc_f1(ytrue=ytrue, ypred=ypred, type="macro")

        # calculate bias
        self.result["average_bias"] = self._calc_bias(
            ypred=ypred,
            sensitive_group_memberships=sensitive_group_memberships,
            num_classes=num_classes,
            num_sensitive_groups=num_sensitive_groups,
        )

        # report classification results per demographic group
        self.result["preds"] = self._get_preds(
            ypred=ypred,
            sensitive_group_memberships=sensitive_group_memberships,
            num_classes=num_classes,
            num_sensitive_groups=num_sensitive_groups,
        )

        return self.result

    @staticmethod
    def _calc_accuracy(ytrue, ypred):
        """
        calculate accracy

        :param ytrue: ground-truth label
        :param ypred: predicted label

        :return: accuracy
        """
        return np.sum(ytrue == ypred) / ytrue.shape[0]

    @staticmethod
    def _calc_f1(ytrue, ypred, type="binary"):
        """
        calculate binary/micro/macro f1 score

        :param ytrue: ground-truth label
        :param ypred: predicted label
        :param type: which f1 score to calculate, binary, micro or macro

        :return: binary/micro/macro f1 score
        """
        return f1_score(ytrue, ypred, average=type)

    def _get_preds(
        self,
        ypred,
        sensitive_group_memberships,
        num_classes,
        num_sensitive_groups,
    ):
        """
        get classification results per demographic group

        :param ypred: predicted label
        :param sensitive_group_memberships: a list of sensitive group indicator for each data point
        :param num_classes: number of class
        :param num_sensitive_groups: number of demographic groups

        :return: classification results per demographic group
        """
        res = {
            i: {j: 0 for j in range(num_classes)} for i in range(num_sensitive_groups)
        }

        num_sample = ypred.shape[0]
        for i in range(num_sample):
            label, sensitive_group_membership = ypred[i], sensitive_group_memberships[i]
            res[round(sensitive_group_membership)][round(label)] += 1

        return res

    def _calc_bias(
        self,
        ypred,
        sensitive_group_memberships,
        num_classes,
        num_sensitive_groups,
    ):
        """
        calculate mean imparity

        :param ypred: predicted label
        :param sensitive_group_memberships: a list of sensitive group indicator for each data point
        :param num_classes: number of class
        :param num_sensitive_groups: number of demographic groups

        :return: mean imparity
        """
        # get classification results
        preds = self._get_preds(
            ypred=ypred,
            sensitive_group_memberships=sensitive_group_memberships,
            num_classes=num_classes,
            num_sensitive_groups=num_sensitive_groups,
        )

        # calculate pairwise imparity between any two demographic groups for each class
        sensitive_groups = list(preds.keys())
        single_imparity = [[] for _ in range(num_classes)]
        for i in range(num_sensitive_groups):
            for j in range(i, num_sensitive_groups):
                m1, m2 = sensitive_groups[i], sensitive_groups[j]
                for y in range(num_classes):
                    if sum(list(preds[m1].values())) == 0:
                        p_y_m1 = 0
                    else:
                        p_y_m1 = preds[m1][y] / sum(list(preds[m1].values()))
                    if sum(list(preds[m2].values())) == 0:
                        p_y_m2 = 0
                    else:
                        p_y_m2 = preds[m2][y] / sum(list(preds[m2].values()))
                    diff = abs(p_y_m1 - p_y_m2)
                    single_imparity[y].append(diff)

        # calculate mean imparity
        single_imparity = [np.mean(x) for x in single_imparity]
        total_imparity = np.mean(single_imparity)
        return total_imparity

    @staticmethod
    def _default_result():
        """
        initialize default dict for reporting evaluation results

        :return: a dict with essential key-value pairs to report results
        """
        return {
            "stage": "test",
            "loss": 0,
            "accuracy": 0,
            "binary_f1": 0,
            "micro_f1": 0,
            "macro_f1": 0,
            "average_bias": 0,
            "preds": {},
        }
