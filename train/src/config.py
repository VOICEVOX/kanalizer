from dataclasses import dataclass


def migrate(config: dict):
    # num_models_to_keepをnum_last_models_to_keepとnum_best_models_to_keepに分割
    if "num_models_to_keep" in config:
        config["num_last_models_to_keep"] = config["num_models_to_keep"]
        config["num_best_models_to_keep"] = config["num_models_to_keep"]
        del config["num_models_to_keep"]

    if "optimizer_lr" not in config:
        config["optimizer_lr"] = 1e-3
    if "exponential_lr_scheduler_gamma" not in config:
        config["exponential_lr_scheduler_gamma"] = 0.9

    if "weight_decay" not in config:
        config["weight_decay"] = 0

    return config


@dataclass
class Config:
    train_data: str
    eval_data: str
    eval_max_words: int
    dim: int
    max_epochs: int
    num_last_models_to_keep: int
    num_best_models_to_keep: int
    seed: int
    optimizer_lr: float
    exponential_lr_scheduler_gamma: float
    weight_decay: float

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**migrate(config))
