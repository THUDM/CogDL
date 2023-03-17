import torch
import torch.nn.functional as F
from cogdl.layers import GCNLayer

from .. import BaseModel
from .gcn import GCN


class GAE(GCN):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_layers, args.dropout)

    def __init__(self, in_feats, hidden_size, num_layers, dropout):
        super(GAE, self).__init__(in_feats, hidden_size, 1, num_layers, dropout)

    def make_loss(self, data, adj):
        embeddings = self.embed(data)
        return (
            F.binary_cross_entropy(F.softmax(torch.mm(embeddings, embeddings.t())), adj, reduction="sum")
            / data.x.shape[0]
        )

    def get_features(self, data):
        return self.embed(data).detach()


class VGAE(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size)

    def __init__(self, num_features, hidden_size):
        super(VGAE, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.conv1 = GCNLayer(self.num_features, self.hidden_size)
        self.conv2_mean = GCNLayer(self.hidden_size, self.hidden_size)
        self.conv2_var = GCNLayer(self.hidden_size, self.hidden_size)

    def reparameterize(self, mean, log_var):
        log_var = log_var.clamp(max=10)
        sigma = torch.exp(log_var)
        z = mean + torch.randn_like(log_var) * sigma
        return z

    def encode(self, graph):
        graph.add_remaining_self_loops()
        graph.sym_norm()

        h = graph.x
        h = self.conv1(graph, h)
        h = F.relu(h)
        mean = self.conv2_mean(graph, h)
        log_var = self.conv2_var(graph, h)
        return mean, log_var

    def decode(self, x):
        return torch.sigmoid(torch.matmul(x, x.t()))

    def forward(self, graph):
        mean, log_var = self.encode(graph)
        return self.reparameterize(mean, log_var)

    def get_features(self, graph):
        return self.forward(graph).detach()

    def make_loss(self, data, adj):
        mean, log_var = self.encode(data)
        z = self.reparameterize(mean, log_var)
        mat = self.decode(z)
        recon_loss = F.binary_cross_entropy(mat, adj, reduction="sum")
        var = torch.exp(log_var)
        kl_loss = 0.5 * torch.mean(torch.sum(mean * mean + var - log_var - 1, dim=1))
        print("recon_loss = %.3f, kl_loss = %.3f" % (recon_loss, kl_loss))
        return recon_loss + kl_loss
