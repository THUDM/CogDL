from cogdl.tasks.attributed_graph_clustering import AttributedGraphClustering
import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict
import unittest

graph_clustering_task_name = "attributed_graph_clustering"


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "device_id": [0],
        "num_clusters": 7,
        "cluster_method": "kmeans",
        "hidden_size": 16,
        "model_type": "emb",
        "momentum": 0,
        "enhance": None,
        "cpu": not cuda_available,
        "checkpoint": False,
    }
    return build_args_from_dict(default_dict)


def create_simple_task():
    args = get_default_args()
    args.task = graph_clustering_task_name
    args.dataset = "cora"
    args.model = "prone"
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    return AttributedGraphClustering(args)


class TestAttributedGraphClustering(unittest.TestCase):
    simple_task = None  # prone cora task
    simple_task_train_result = None

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.simple_task:
            cls.simple_task = create_simple_task()
            cls.simple_task_train_result = cls.simple_task.train()

    def test_accuracy_within_range(self):
        assert 0 <= self.__class__.simple_task_train_result["Accuracy"] <= 1

    def test_micro_f1_within_range(self):
        assert 0 <= self.__class__.simple_task_train_result["Micro_F1"] <= 1

    def test_nmi_within_range(self):
        assert 0 <= self.__class__.simple_task_train_result["NMI"] <= 1

    def test_all_metrics_present(self):
        assert "Accuracy" in self.__class__.simple_task_train_result
        assert "NMI" in self.__class__.simple_task_train_result
        assert "Micro_F1" in self.__class__.simple_task_train_result

    def test_correct_class_resolved_from_build_task(self):
        args = get_default_args()
        args.task = graph_clustering_task_name
        args.dataset = "cora"
        args.model = "prone"
        args.step = 5
        args.theta = 0.5
        args.mu = 0.2
        assert isinstance(build_task(args), AttributedGraphClustering)


if __name__ == "__main__":
    unittest.main()
