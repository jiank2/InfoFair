class ModelConfigs:
    def __init__(self):
        self.configs = {
            "adult": {
                "feature_extractor": [32],
                "classifier": [],
                "sensitive_classifier": []
            },
        }

    def get_configs(self):
        """
        return the default model configs

        :return: a dict of model configs
        """
        return self.configs
