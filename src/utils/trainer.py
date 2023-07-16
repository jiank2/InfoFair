import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.evaluator import Evaluator
from utils.mmd_loss import MMDLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        configs,
        model_config,
        data,
        model,
        device,
    ):
        # load configs
        self.configs = self._default_configs()
        self.configs.update(configs)
        self._update_save_path()

        self.model_config = model_config
        self.device = device

        # load data and model
        self.data = data
        self.model = model
        self.model.to(device)

        # initialize settings
        self.opt = torch.optim.Adam(
            model.parameters(),
            lr=self.configs["lr"],
            weight_decay=self.configs["weight_decay"],
        )

        if self.configs["loss"] == "nll":
            self.clf_crit = nn.NLLLoss()
        elif self.configs["loss"] == "cross_entropy":
            self.clf_crit = nn.CrossEntropyLoss()

        if self.configs["sensitive_loss"] == "nll":
            self.sensitive_crit = nn.NLLLoss()
        elif self.configs["sensitive_loss"] == "cross_entropy":
            self.sensitive_crit = nn.CrossEntropyLoss()
        elif self.configs["sensitive_loss"] == "mmd":
            self.sensitive_crit = MMDLoss(bandwidth=100.0)

        self._reset_patience()
        self._reset_temperature()

        # initialize evaluator
        self.evaluator = Evaluator(fairness=self.configs["fairness"], logger=logger)

        # get dataloaders
        self.loader = {
            "train": self.data.train_loader,
            "validation": self.data.val_loader,
            "test": self.data.test_loader,
        }

        # misc
        self.min_val_loss = float("inf")

    def train(self):
        """
        train model
        """
        for epoch in range(self.configs["num_epochs"]):
            write_to_file = False
            logger.info("Epoch: {epoch}".format(epoch=epoch))

            # temperature annealing for gumbel softmax
            if (epoch % 50 == 0) and (epoch != 0):
                self.temperature /= 2.0

            # train one epoch
            self._train_epoch()

            # evaluate on training data
            self._test_epoch(stage="train", write_to_file=write_to_file)

            # if reach max #epochs, write to file
            if epoch == self.configs["num_epochs"] - 1:
                write_to_file = True

            # evaluate on validation data
            self._test_epoch(stage="validation", write_to_file=write_to_file)

            # if early stopping, write to file
            if self.patience == 0:
                write_to_file = True
                # evaluate on validation data
                self._test_epoch(stage="validation", write_to_file=write_to_file)
                return

    def test(self):
        """
        evaluate on test data
        """
        self._test_epoch(stage="test", write_to_file=True)

    def _train_epoch(self):
        """
        train model for ***one epoch only***
        """
        self.model.train()
        for batch_idx, (feature, label, sensitive_label) in enumerate(
            self.data.train_loader
        ):
            self.opt.zero_grad()

            feature = feature.to(self.device)
            label = label.to(self.device)
            sensitive_label = sensitive_label.to(self.device)

            class_log_prob, sensitive_log_prob, density_ratio_mean = self.model(
                feature=feature,
                sensitive_label=sensitive_label,
                temperature=self.temperature,
                is_hard_gumbel_softmax=False,
                device=self.device,
            )

            pred_loss = self.clf_crit(class_log_prob, label)
            adversarial_loss = self.configs["regularization"] * self.sensitive_crit(
                sensitive_log_prob, sensitive_label
            )
            density_ratio_loss = self.configs["regularization"] * density_ratio_mean
            loss = pred_loss - adversarial_loss + density_ratio_loss

            loss.backward()
            self.opt.step()

    def _test_epoch(self, stage, write_to_file=False):
        """
        evaluate model for each epoch

        :param stage: train, validation or test
        :param write_to_file: a flag indicating write evaluation results to file or not
        """
        self.model.eval()
        label_preds, label_truth, sensitive_truth = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )
        loss, num_test = 0, 0
        for batch_idx, (feature, label, sensitive_label) in enumerate(
            self.loader[stage]
        ):
            num_sample = feature.size()[0]
            num_test += num_sample

            # get results for each batch of data
            class_log_prob, sensitive_log_prob, density_ratio_mean = self.model(
                feature=feature,
                sensitive_label=sensitive_label,
                temperature=self.temperature,
                is_hard_gumbel_softmax=False,
                device=self.device,
            )

            # calculate loss for target prediction
            pred_loss = self.clf_crit(class_log_prob, label)
            # calculate loss for sensitive attribute prediction
            adversarial_loss = self.configs["regularization"] * self.sensitive_crit(
                sensitive_log_prob, sensitive_label
            )
            # calculate loss for density ratio estimation
            density_ratio_loss = self.configs["regularization"] * density_ratio_mean
            # calculate overall loss
            loss = (
                loss + (pred_loss - adversarial_loss + density_ratio_loss) * num_sample
            )

            # get predicted labels, ground-truth labels and ground-truth demographic group memberships
            ypred = class_log_prob.max(dim=1).indices
            label_preds = ypred if batch_idx == 0 else torch.cat((label_preds, ypred))
            label_truth = label if batch_idx == 0 else torch.cat((label_truth, label))
            sensitive_truth = (
                sensitive_label
                if batch_idx == 0
                else torch.cat((sensitive_truth, sensitive_label))
            )

        # calculate empirical loss for all test points
        loss /= num_test

        if stage == "validation" and loss >= self.min_val_loss:
            self.patience -= 1
        elif stage == "validation" and loss < self.min_val_loss:
            self.min_val_loss = loss
            self._reset_patience()

        # evaluate
        result = self.evaluator.eval(
            loss=loss,
            ypred=label_preds,
            ytrue=label_truth,
            sensitive_group_memberships=sensitive_truth,
            num_classes=self.data.num_classes,
            num_sensitive_groups=self.data.num_sensitive_groups,
            stage=stage,
        )

        # dict to string for printing
        result_str = self._result_dict_to_print_str(stage=stage, result=result)
        logger.info(result_str)

        if write_to_file:
            self._write_to_file(result)

    @staticmethod
    def _result_dict_to_print_str(stage, result):
        """
        dict to string for printing

        :param stage: training, validation or test
        :param result: a dict of evaluation results

        :return: a string with all evaluation results
        """
        result_str = "{stage} - ".format(stage=stage)
        for metric in (
            "loss",
            "accuracy",
            "binary_f1",
            "micro_f1",
            "macro_f1",
            "average_bias",
        ):
            result_str += "{key}: {value}\t".format(key=metric, value=result[metric])
        result_str = result_str[:-1] + "\n\t\t"
        result_str += "{key}: {value}".format(key="preds", value=result["preds"])
        return result_str

    def _write_to_file(self, res):
        """
        write result to file

        :param res: a string with all evaluation results
        """
        path = "../result/{nlayer}-layer/{path}".format(
            nlayer=len(self.model_config["feature_extractor"]),
            path=self.configs["save_path"],
        )
        folder = "/".join(path.split("/")[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)

        print("save to file: {path}".format(path=path))
        res_str = ""
        with open(path, "a") as f:
            res["model"] = self.configs["model"]
            for k in res:
                res_str += "{k}: {v}\t".format(k=k, v=res[k])
            f.write(
                "{res}\n".format(
                    # res=json.dumps(res)
                    res=res_str[:-1]
                )
            )
            f.flush()

    def _reset_patience(self):
        """
        reset patience for early stopping
        """
        self.patience = self.configs["patience"]

    def _reset_temperature(self):
        """
        reset temperature of gumbel softmax
        """
        self.temperature = self.configs["temperature"]

    @staticmethod
    def _default_configs():
        """
        get initial training config
        """
        configs = {
            "model": "infofair",
            "dataset": "adult",
            "sensitive": "sex",
            "fairness": "statistical_parity",
            "loss": "nll",
            "sensitive_loss": "nll",
            "num_epochs": 100,
            "patience": 5,
            "regularization": 0.1,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "dropout": 0,
            "temperature": 1.0,
            "seed": 0,
        }
        return configs

    def _update_save_path(self):
        """
        get path to save evaluation results
        """
        # get dir name for sensitive attribute
        sensitive_dir = self.configs["sensitive"].strip().split(",")
        sensitive_dir = " ".join(sensitive_dir)

        # get file name
        filename = "lambda={lambda_}_lr={lr}_weight-decay={weight_decay}_dropout={dropout}_seed={seed}.txt".format(
            lambda_=self.configs["regularization"],
            lr=self.configs["lr"],
            weight_decay=self.configs["weight_decay"],
            dropout=self.configs["dropout"],
            seed=self.configs["seed"],
        )

        # get path for saving evaluation results
        self.configs["save_path"] = os.path.join(
            self.configs["fairness"],
            self.configs["dataset"],
            sensitive_dir,
            filename,
        )
