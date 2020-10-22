from typing import Any, Union, Type, Optional

from cogdl.models import register_model
from cogdl.models.supervised_model import (
    SupervisedHomogeneousNodeClassificationModel,
    SupervisedHeterogeneousNodeClassificationModel,
)

from cogdl.trainers.gpt_gnn_trainer import (
    GPT_GNNHomogeneousTrainer,
    GPT_GNNHeterogeneousTrainer,
)


#
# @register_model("gpt_gnn")
# class GPT_GNN(BaseModel):
#     def __init__(
#         self,
#         in_dim,
#         n_hid,
#         num_types,
#         num_relations,
#         n_heads,
#         n_layers,
#         dropout=0.2,
#         conv_name="hgt",
#         prev_norm=False,
#         last_norm=False,
#         use_RTE=True,
#     ):
#         super(GPT_GNN, self).__init__()
#         self.gcs = nn.ModuleList()
#         self.num_types = num_types
#         self.in_dim = in_dim
#         self.n_hid = n_hid
#         self.adapt_ws = nn.ModuleList()
#         self.drop = nn.Dropout(dropout)
#         for t in range(num_types):
#             self.adapt_ws.append(nn.Linear(in_dim, n_hid))
#         for l in range(n_layers - 1):
#             self.gcs.append(
#                 GeneralConv(
#                     conv_name,
#                     n_hid,
#                     n_hid,
#                     num_types,
#                     num_relations,
#                     n_heads,
#                     dropout,
#                     use_norm=prev_norm,
#                     use_RTE=use_RTE,
#                 )
#             )
#         self.gcs.append(
#             GeneralConv(
#                 conv_name,
#                 n_hid,
#                 n_hid,
#                 num_types,
#                 num_relations,
#                 n_heads,
#                 dropout,
#                 use_norm=last_norm,
#                 use_RTE=use_RTE,
#             )
#         )
#
#     def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
#         res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
#         for t_id in range(self.num_types):
#             idx = node_type == int(t_id)
#             if idx.sum() == 0:
#                 continue
#             res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
#         meta_xs = self.drop(res)
#         del res
#         for gc in self.gcs:
#             meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
#         return meta_xs


@register_model("gpt_gnn")
class GPT_GNN(
    SupervisedHomogeneousNodeClassificationModel,
    SupervisedHeterogeneousNodeClassificationModel,
):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        """
            Dataset arguments
        """
        parser.add_argument(
            "--use_pretrain", help="Whether to use pre-trained model", action="store_true"
        )
        parser.add_argument(
            "--pretrain_model_dir",
            type=str,
            default="/datadrive/models/gpt_all_cs",
            help="The address for pretrained model.",
        )
        # parser.add_argument(
        #     "--model_dir",
        #     type=str,
        #     default="/datadrive/models/gpt_all_reddit",
        #     help="The address for storing the models and optimization results.",
        # )
        parser.add_argument(
            "--task_name",
            type=str,
            default="reddit",
            help="The name of the stored models and optimization results.",
        )
        parser.add_argument(
            "--sample_depth", type=int, default=6, help="How many numbers to sample the graph"
        )
        parser.add_argument(
            "--sample_width",
            type=int,
            default=128,
            help="How many nodes to be sampled per layer per type",
        )
        """
           Model arguments 
        """
        parser.add_argument(
            "--conv_name",
            type=str,
            default="hgt",
            choices=["hgt", "gcn", "gat", "rgcn", "han", "hetgnn"],
            help="The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)",
        )
        parser.add_argument("--n_hid", type=int, default=400, help="Number of hidden dimension")
        parser.add_argument("--n_heads", type=int, default=8, help="Number of attention head")
        parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers")
        parser.add_argument(
            "--prev_norm",
            help="Whether to add layer-norm on the previous layers",
            action="store_true",
        )
        parser.add_argument(
            "--last_norm",
            help="Whether to add layer-norm on the last layers",
            action="store_true",
        )
        parser.add_argument("--dropout", type=int, default=0.2, help="Dropout ratio")

        """
            Optimization arguments
        """
        parser.add_argument(
            "--optimizer",
            type=str,
            default="adamw",
            choices=["adamw", "adam", "sgd", "adagrad"],
            help="optimizer to use.",
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            help="Name of learning rate scheduler.",
            choices=["cycle", "cosine"],
        )
        parser.add_argument(
            "--data_percentage",
            type=int,
            default=0.1,
            help="Percentage of training and validation data to use",
        )
        parser.add_argument("--n_epoch", type=int, default=50, help="Number of epoch to run")
        parser.add_argument(
            "--n_pool", type=int, default=8, help="Number of process to sample subgraph"
        )
        parser.add_argument(
            "--n_batch",
            type=int,
            default=10,
            help="Number of batch (sampled graphs) for each epoch",
        )
        parser.add_argument(
            "--batch_size", type=int, default=64, help="Number of output nodes for training"
        )
        parser.add_argument("--clip", type=int, default=0.5, help="Gradient Norm Clipping")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return GPT_GNN()

    def loss(self, data: Any) -> Any:
        pass

    def predict(self, data: Any) -> Any:
        pass

    def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
        pass

    @staticmethod
    def get_trainer(
        taskType: Any, args
    ) -> Optional[Type[Union[GPT_GNNHomogeneousTrainer, GPT_GNNHeterogeneousTrainer]]]:
        # if taskType == NodeClassification:
        return GPT_GNNHomogeneousTrainer
        # elif taskType == HeterogeneousNodeClassification:
        #     return GPT_GNNHeterogeneousTrainer
        # else:
        #     return None
