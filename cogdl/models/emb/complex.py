import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .knowledge_base import KGEModel


@register_model("complex")
class ComplEx(KGEModel):
    r"""
    the implementation of ComplEx model from the paper `"Complex Embeddings for Simple Link Prediction"<http://proceedings.mlr.press/v48/trouillon16.pdf>`
    borrowed from `KnowledgeGraphEmbedding<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>`
    """

    def score(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score
