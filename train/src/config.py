from dataclasses import dataclass


def migrate(config: dict):
    # num_models_to_keepをnum_last_models_to_keepとnum_best_models_to_keepに分割
    if "num_models_to_keep" in config:
        config["num_last_models_to_keep"] = config["num_models_to_keep"]
        config["num_best_models_to_keep"] = config["num_models_to_keep"]
        del config["num_models_to_keep"]

    if "optimizer_lr" not in config:
        config["optimizer_lr"] = 1e-3

    if "exponential_lr_scheduler_gamma" in config:
        del config["exponential_lr_scheduler_gamma"]

    if "weight_decay" not in config:
        config["weight_decay"] = 0

    if "test_ratio" not in config:
        config["test_ratio"] = 0

    if "use_layernorm" in config:
        del config["use_layernorm"]

    return config


@dataclass
class Config:
    train_data: str
    test_ratio: float
    eval_data: str
    eval_max_words: int
    dim: int
    max_epochs: int
    num_last_models_to_keep: int
    num_best_models_to_keep: int
    seed: int
    optimizer_lr: float
    weight_decay: float

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**migrate(config))
