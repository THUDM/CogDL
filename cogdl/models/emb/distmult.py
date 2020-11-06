import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .knowledge_base import KGEModel

@register_model("distmult")
class DistMult(KGEModel):
    r"""The DistMult model from the ICLR 2015 paper `"EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES"
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf>`
    borrowed from `KnowledgeGraphEmbedding<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>`
    """

    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(DistMult, self).__init__(nentity, nrelation, hidden_dim, gamma, double_entity_embedding, double_relation_embedding)

    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score