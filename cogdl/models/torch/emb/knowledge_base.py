import torch
import torch.nn as nn

from .. import BaseModel


class KGEModel(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--embedding_size", type=int, default=500, help="Dimensionality of embedded vectors")
        parser.add_argument("--gamma", type=float,default=12.0, help="Hyperparameter for embedding")
        parser.add_argument("--double_entity_embedding", action="store_true")
        parser.add_argument("--double_relation_embedding", action="store_true")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_entities,
            args.num_rels,
            args.embedding_size,
            args.gamma,
            args.double_entity_embedding,
            args.double_relation_embedding,
        )

    def __init__(
        self, nentity, nrelation, hidden_dim, gamma, double_entity_embedding, double_relation_embedding
    ):

        super(KGEModel, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False
        )
        
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def forward(self, sample, mode="single"):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == "single":
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )

        else:
            raise ValueError("mode %s not supported" % mode)

        score = self.score(head, relation, tail, mode)

        return score

    def score(self, head, relation, tail, mode):
        raise NotImplementedError
