import importlib

from .base_trainer import BaseTrainer


TRAINER_REGISTRY = {}


def register_trainer(name):
    """
    New universal trainer types can be added to cogdl with the :func:`register_trainer`
    function decorator.

    For example::

        @register_trainer('self_auxiliary_task')
        class SelfAuxiliaryTaskTrainer(BaseModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate universal trainer ({})".format(name))
        if not issubclass(cls, BaseTrainer):
            raise ValueError("Model ({}: {}) must extend BaseTrainer".format(name, cls.__name__))
        TRAINER_REGISTRY[name] = cls
        cls.trainer_name = name
        return cls

    return register_trainer_cls


def try_import_trainer(trainer):
    if trainer not in TRAINER_REGISTRY:
        if trainer in SUPPORTED_TRAINERS:
            importlib.import_module(SUPPORTED_TRAINERS[trainer])
        else:
            print(f"Failed to import {trainer} trainer.")
            return False
    return True


def build_trainer(args):
    if not try_import_trainer(args.trainer):
        exit(1)
    return TRAINER_REGISTRY[args.trainer].build_trainer_from_args(args)


SUPPORTED_TRAINERS = {
    "graphsaint": "cogdl.trainers.sampled_trainer",
    "neighborsampler": "cogdl.trainers.sampled_trainer",
    "clustergcn": "cogdl.trainers.sampled_trainer",
    "random_cluster": "cogdl.trainers.sampled_trainer",
    "self_supervised_pt_ft": "cogdl.trainers.self_supervised_trainer",
    "self_supervised_joint": "cogdl.trainers.self_supervised_trainer",
    "m3s": "cogdl.trainers.m3s_trainer",
    "distributed_trainer": "cogdl.trainers.distributed_trainer",
    "dist_clustergcn": "cogdl.trainers.distributed_sampled_trainer",
    "dist_neighborsampler": "cogdl.trainers.distributed_sampled_trainer",
    "dist_saint": "cogdl.trainers.distributed_sampled_trainer",
}
