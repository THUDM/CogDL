from abc import ABC, abstractmethod
import torch


class BaseTrainer(ABC):
    def __init__(self, args=None):
        if args is not None:
            device_id = args.device_id if hasattr(args, "device_id") else [0]
            self.device = (
                "cpu" if not torch.cuda.is_available() or (hasattr(args, "cpu") and args.cpu) else device_id[0]
            )
            self.patience = args.patience if hasattr(args, "patience") else 10
            self.max_epoch = args.max_epoch if hasattr(args, "max_epoch") else 100
            self.lr = args.lr
            self.weight_decay = args.weight_decay
            self.loss_fn, self.evaluator = None, None
            self.data, self.train_loader, self.optimizer = None, None, None
            self.num_workers = args.num_workers if hasattr(args, "num_workers") else 0

    @classmethod
    @abstractmethod
    def build_trainer_from_args(cls, args):
        """Build a new trainer instance."""
        raise NotImplementedError("Trainers must implement the build_trainer_from_args method")

    @abstractmethod
    def fit(self, model, dataset):
        raise NotImplementedError
