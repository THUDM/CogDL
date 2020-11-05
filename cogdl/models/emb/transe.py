import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .knowledge_base import KGEModel

@register_model("transe")
class TransE(KGEModel):
    r"""The TransE model from paper `"Translating Embeddings for Modeling Multi-relational Data"
    <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`
    borrowed from `KnowledgeGraphEmbedding<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>`
    """
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(TransE, self).__init__(nentity, nrelation, hidden_dim, gamma, True, True)

    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score