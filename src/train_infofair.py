import argparse

import numpy as np
import torch
import torch.multiprocessing as mp

from model_configs import ModelConfigs
from models.infofair import InfoFair
from models.vanilla import Vanilla
from train_configs import TrainConfigs
from utils.data_loader import Dataset
from utils.trainer import Trainer

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--enable_cuda", action="store_true", default=True, help="Enable CUDA training."
)
parser.add_argument(
    "--device_number", type=int, default=0, help="Which GPU to use for CUDA training."
)
parser.add_argument("--dataset", type=str, default="adult", help="Dataset to train.")
parser.add_argument(
    "--sensitive", type=str, default="sex,race", help="Sensitive attribute(s)."
)
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of epochs to train."
)
parser.add_argument(
    "--patience", type=int, default=5, help="Patience for early stopping."
)
parser.add_argument(
    "--regularization", type=float, default=0.1, help="Regularization hyperparameter."
)
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="Weight decay.",
)
parser.add_argument(
    "--dropout", type=float, default=0.0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature annealing for gumbel softmax.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--train_batch_size", type=int, default=128, help="Batch size for training."
)

args = parser.parse_args()
use_cuda = args.enable_cuda and torch.cuda.is_available()
if use_cuda:
    args.device_name = f"cuda:{args.device_number}"
else:
    args.device_name = "cpu"


def train(model_config, train_config, preferred_label=1, device_name="cuda:0"):
    """
    function to train infofair

    :param model_config: a dict of model configs
    :param train_config: a dict of training configs
    :param preferred_label: (***not implemented yet***) a preferred label for equal opportunity
    :param device_name: GPU to train model
    """
    # set random seed
    if train_config["seed"] is not None:
        np.random.seed(train_config["seed"])
        torch.manual_seed(train_config["seed"])
        torch.cuda.manual_seed(train_config["seed"])

    # create dataset
    data = Dataset(
        dataset=train_config["dataset"], device_name=train_config["device_name"]
    )
    data.create_sensitive_features(sensitive_attr=train_config["sensitive"])

    # initialize InfoFair
    model = InfoFair(
        nfeat=data.features.size()[1],
        nhids=model_config,
        nclass=data.num_classes,
        nsensitive=data.num_sensitive_groups,
        droprate=train_config["dropout"],
    )

    # create dataloader
    data.create_dataloader(train_batch_size=train_config["train_batch_size"])

    # initialize trainer
    trainer = Trainer(
        train_config,
        model_config,
        data,
        model,
        torch.device(train_config["device_name"]),
    )

    # train model
    trainer.train()

    # test model
    trainer.test()


if __name__ == "__main__":
    # get initial configs
    train_config = TrainConfigs()
    train_config = train_config.get_configs()
    train_config.update(vars(args))

    model_configs = ModelConfigs()
    model_configs = model_configs.get_configs()

    # select model configs
    model_config = model_configs[args.dataset]

    # initialize training configs
    train_config["model"] = "infofair"  # model name
    train_config["fairness"] = "statistical_parity"  # fairness definition
    train_config["seed"] = 4  # random seed
    train_config["lr"] = 0.0001  # learning rate
    train_batch_size = 128

    train(
        model_config=model_config,
        train_config=train_config,
        preferred_label=1,
        device_name="cuda:3",  # if only 1 gpu, change to cuda:0
    )
