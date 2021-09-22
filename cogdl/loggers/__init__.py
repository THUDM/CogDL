from .base_logger import Logger


def build_logger(logger, log_path="./runs", project="cogdl-exp"):
    if logger == "wandb":
        from .wandb_logger import WandbLogger

        return WandbLogger(log_path, project)
    elif logger == "tensorboard":
        from .tensorboard_logger import TBLogger

        return TBLogger(log_path)
    else:
        return Logger(log_path)
