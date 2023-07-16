class TrainConfigs:
    def __init__(self):
        self.configs = {
            "model": "infofair",
            "dataset": "adult",
            "sensitive": "sex",
            "fairness": "statistical_parity",
            "preferred_label": 1,  # only use for equal opportunity (not implemented yet)
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

    def get_configs(self):
        """
        return the default training configs

        :return: a dict of training configs
        """
        return self.configs
