import importlib
import os

from .base_task import BaseTask

TASK_REGISTRY = {}


def register_task(name):
    """
    New task types can be added to cogdl with the :func:`register_task`
    function decorator.

    For example::

        @register_task('node_classification')
        class NodeClassification(BaseTask):
            (...)

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, BaseTask):
            raise ValueError("Task ({}: {}) must extend BaseTask".format(name, cls.__name__))
        TASK_REGISTRY[name] = cls
        return cls

    return register_task_cls


def try_import_task(task):
    if task not in TASK_REGISTRY:
        if task in SUPPORTED_TASKS:
            importlib.import_module(SUPPORTED_TASKS[task])
        else:
            print(f"Failed to import {task}.")
            return False
    return True


def build_task(args, dataset=None, model=None):
    if not try_import_task(args.task):
        exit(1)
    if dataset is None and model is None:
        return TASK_REGISTRY[args.task](args)
    elif dataset is not None and model is None:
        return TASK_REGISTRY[args.task](args, dataset=dataset)
    elif dataset is None and model is not None:
        return TASK_REGISTRY[args.task](args, model=model)
    return TASK_REGISTRY[args.task](args, dataset=dataset, model=model)


SUPPORTED_TASKS = {
    "attributed_graph_clustering": "cogdl.tasks.attributed_graph_clustering",
    "graph_classification": "cogdl.tasks.graph_classification",
    "heterogeneous_node_classification": "cogdl.tasks.heterogeneous_node_classification",
    "link_prediction": "cogdl.tasks.link_prediction",
    "multiplex_link_prediction": "cogdl.tasks.multiplex_link_prediction",
    "multiplex_node_classification": "cogdl.tasks.multiplex_node_classification",
    "node_classification": "cogdl.tasks.node_classification",
    "oag_supervised_classification": "cogdl.tasks.oag_supervised_classification",
    "oag_zero_shot_infer": "cogdl.tasks.oag_zero_shot_infer",
    "pretrain": "cogdl.tasks.pretrain",
    "similarity_search": "cogdl.tasks.similarity_search",
    "unsupervised_graph_classification": "cogdl.tasks.unsupervised_graph_classification",
    "unsupervised_node_classification": "cogdl.tasks.unsupervised_node_classification",
    "recommendation": "cogdl.tasks.recommendation",
}
