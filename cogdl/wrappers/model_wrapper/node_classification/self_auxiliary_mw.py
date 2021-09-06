import torch
from .. import ModelWrapper, register_model_wrapper
from cogdl.models.nn.self_auxiliary_task import (
    EdgeMask,
    PairwiseDistance,
    PairwiseAttrSim,
    AttributeMask,
    Distance2Clusters,
)
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_logreg


@register_model_wrapper("self_auxiliary_mw")
class SelfAuxiliaryTask(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--auxiliary-task", type=str, default="edge_mask",
                            help="Option: edge_mask, attribute_mask, distance2clusters,"
                                 " pairwise_distance, pairwise_attr_sim")
        parser.add_argument("--dropedge-rate", type=float, default=0.0)
        parser.add_argument("--mask-ratio", type=float, default=0.1)
        # fmt: on

    def __init__(self, model, optimizer_cfg, auxiliary_task, dropedge_rate, mask_ratio):
        super().__init__()
        self.auxiliary_task = auxiliary_task
        self.optimizer_cfg = optimizer_cfg
        self.hidden_size = optimizer_cfg["hidden_size"]
        self.dropedge_rate = dropedge_rate
        self.mask_ratio = mask_ratio
        self.model = model

        self.agent = None

    def train_step(self, subgraph):
        graph = subgraph
        with graph.local_graph():
            graph = self.agent.transform_data(graph)
            pred = self.model(graph)
        sup_loss = self.default_loss_fn(pred, graph.y)
        ssl_loss = self.agent.make_loss(pred)
        return sup_loss + ssl_loss

    # def evaluate(self, dataset):
    #     graph = dataset.data.to(self.device)

    def test_step(self, graph):
        with torch.no_grad():
            pred = self.model(graph)
        y = graph.y
        result = evaluate_node_embeddings_using_logreg(pred, y, graph.train_mask, graph.test_mask)
        self.note("test_acc", result)

    def generate_virtual_labels(self, data):
        if self.auxiliary_task == "edge_mask":
            self.agent = EdgeMask(self.hidden_size, self.mask_ratio, self.device)
        elif self.auxiliary_task == "attribute_mask":
            self.agent = AttributeMask(data, self.hidden_size, data.train_mask, self.mask_ratio, self.device)
        elif self.auxiliary_task == "pairwise_distance":
            self.agent = PairwiseDistance(
                self.hidden_size,
                [(1, 2), (2, 3), (3, 5)],
                self.sampling,
                self.dropedge_rate,
                256,
                self.device,
            )
        elif self.auxiliary_task == "distance2clusters":
            self.agent = Distance2Clusters(self.hidden_size, 30, self.device)
        elif self.auxiliary_task == "pairwise_attr_sim":
            self.agent = PairwiseAttrSim(self.hidden_size, 5, self.device)
        else:
            raise Exception(
                "auxiliary task must be edge_mask, attribute_mask, pairwise_distance, distance2clusters,"
                "or pairwise-attr-sim"
            )

    def self_supervised_loss(self, data):
        embed = self.gcn.embed(data)
        return self.agent.make_loss(embed)

    def setup_optimizer(self):
        lr, wd = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
