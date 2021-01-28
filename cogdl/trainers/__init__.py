import importlib

from .base_trainer import BaseTrainer


UNIVERSAL_TRAINER_REGISTRY = {}


def register_universal_trainer(name):
    """
    New universal trainer types can be added to cogdl with the :func:`register_universal_trainer`
    function decorator.

    For example::

        @register_universal_trainer('self_auxiliary_task')
        class SelfAuxiliaryTaskTrainer(BaseModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_universal_trainer_cls(cls):
        if name in UNIVERSAL_TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate universal trainer ({})".format(name))
        if not issubclass(cls, BaseTrainer):
            raise ValueError("Model ({}: {}) must extend BaseTrainer".format(name, cls.__name__))
        UNIVERSAL_TRAINER_REGISTRY[name] = cls
        cls.trainer_name = name
        return cls

    return register_universal_trainer_cls


def try_import_universal_trainer(trainer):
    if trainer not in UNIVERSAL_TRAINER_REGISTRY:
        if trainer in SUPPORTED_UNIVERSAL_TRAINERS:
            importlib.import_module(SUPPORTED_UNIVERSAL_TRAINERS[trainer])
        else:
            print(f"Failed to import {trainer} trainer.")
            return False
    return True


def build_universal_trainer(args):
    if not try_import_universal_trainer(args.trainer):
        exit(1)
    return UNIVERSAL_TRAINER_REGISTRY[args.trainer].build_trainer_from_args(args)


SUPPORTED_UNIVERSAL_TRAINERS = {
    "saint": "cogdl.trainers.sampled_trainer",
    "self_auxiliary_task": "cogdl.trainers.self_auxiliary_task_trainer",
}
