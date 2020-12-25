import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .knowledge_base import KGEModel


@register_model("rotate")
class RotatE(KGEModel):
    r"""
    Implementation of RotatE model from the paper `"RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
    <https://openreview.net/forum?id=HkgEQnRqYQ>`.
    borrowed from `KnowledgeGraphEmbedding<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>`
    """

    def __init__(
        self, nentity, nrelation, hidden_dim, gamma, double_entity_embedding=False, double_relation_embedding=False
    ):
        super(RotatE, self).__init__(nentity, nrelation, hidden_dim, gamma, True, double_relation_embedding)

    def score(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score
