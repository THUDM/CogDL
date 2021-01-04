from cogdl.tasks.attributed_graph_clustering import AttributedGraphClustering
import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict
import unittest

graph_clustering_task_name = "attributed_graph_clustering"


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "task" : graph_clustering_task_name,
        "device_id": [0],
        "num_clusters": 7,
        "cluster_method": "kmeans",
        "hidden_size": 16,
        "model_type": "spectral",
        "enhance": None,
        "cpu": not cuda_available,
        "step": 5,
        "theta" : 0.5,
        "mu" : 0.2,
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

def test_kmeans_cora():
    args = get_default_args()
    args.model_type = "content"
    args.model = "prone"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_spectral_cora():
    args = get_default_args()
    args.model_type = "content"
    args.model = "prone"
    args.dataset = "cora"
    args.cluster_method = "spectral"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_prone_cora():
    args = get_default_args()
    args.model = "prone"
    args.model_type = "spectral"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_prone_citeseer():
    args = get_default_args()
    args.model = "prone"
    args.model_type = "spectral"
    args.dataset = "citeseer"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_prone_pubmed():
    args = get_default_args()
    args.model = "prone"
    args.model_type = "spectral"
    args.dataset = "pubmed"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_deepwalk_cora():
    args = get_default_args()
    args.model = "deepwalk"
    args.model_type = "spectral"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_deepwalk_citeseer():
    args = get_default_args()
    args.model = "deepwalk"
    args.model_type = "spectral"
    args.dataset = "citeseer"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_netmf_cora():
    args = get_default_args()
    args.model = "netmf"
    args.model_type = "spectral"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_netmf_citeseer():
    args = get_default_args()
    args.model = "netmf"
    args.model_type = "spectral"
    args.dataset = "citeseer"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_agc_cora():
    args = get_default_args()
    args.model = "agc"
    args.model_type = "spectral"
    args.dataset = "cora"
    args.cluster_method = "spectral"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_agc_citeseer():
    args = get_default_args()
    args.model = "agc"
    args.model_type = "spectral"
    args.dataset = "citeseer"
    args.cluster_method = "spectral"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_daegc_cora():
    args = get_default_args()
    args.model = "daegc"
    args.model_type = "spectral"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

def test_daegc_cora():
    args = get_default_args()
    args.model = "daegc"
    args.model_type = "spectral"
    args.dataset = "citeseer"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["Accuracy"] > 0

if __name__ == "__main__":
    #unittest.main()
    test_kmeans_cora()
    test_spectral_cora()

    test_prone_cora()
    test_prone_citeseer()
    #test_prone_pubmed()

    test_deepwalk_cora()
    test_deepwalk_citeseer()

    test_netmf_cora()
    test_netmf_citeseer()

    test_agc_cora()
    test_agc_citeseer()

    test_daegc_cora()
    test_daegc_citeseer()
