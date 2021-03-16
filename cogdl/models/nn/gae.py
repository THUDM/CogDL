import torch
import torch.nn.functional as F

from .gcn import TKipfGCN, GraphConvolution
from .. import register_model, BaseModel
from cogdl.trainers.gae_trainer import GAETrainer
from cogdl.utils import add_remaining_self_loops, symmetric_normalization

@register_model("gae")
class GAE(TKipfGCN):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_layers, args.dropout)

    def __init__(self, in_feats, hidden_size, num_layers, dropout):
        super(GAE, self).__init__(in_feats, hidden_size, 1, num_layers, dropout)
        
    def make_loss(self, data):
        embeddings = self.get_embeddings(data.x, data.edge_index)
        adj = torch.sparse_coo_tensor(
            data.edge_index, torch.ones(data.edge_index.shape[1]), torch.Size([data.x.shape[0], data.x.shape[0]])
        ).to_dense()
        return F.binary_cross_entropy(F.softmax(torch.mm(embeddings, embeddings.t())), adj, reduction="sum") / data.x.shape[0]
    
    def get_features(self, data):
        return self.get_embeddings(data.x, data.edge_index).detach()

    def get_trainer(self, task, args):
        return GAETrainer

@register_model("vgae")
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
        self.conv1 = GraphConvolution(self.num_features, self.hidden_size)
        self.conv2_mean = GraphConvolution(self.hidden_size, self.hidden_size)
        self.conv2_var = GraphConvolution(self.hidden_size, self.hidden_size)

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(log_var)
        z = mean + torch.randn_like(log_var) * sigma
        return z

    def encode(self, x, edge_index):
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_attr)

        h = x
        h = self.conv1(h, edge_index, edge_attr)
        h = F.relu(h)
        mean = self.conv2_mean(h, edge_index, edge_attr)
        log_var = self.conv2_var(h, edge_index, edge_attr)
        return mean, log_var

    def decode(self, x):
        return torch.sigmoid(torch.matmul(x, x.t()))

    def forward(self, x, edge_index):
        mean, log_var = self.encode(x, edge_index)
        return self.reparameterize(mean, log_var)

    def get_features(self, data):
        return self.forward(data.x, data.edge_index).detach()

    def make_loss(self, data):
        mean, log_var = self.encode(data.x, data.edge_index)
        z = self.reparameterize(mean, log_var)
        mat = self.decode(z)
        adj = torch.sparse_coo_tensor(
            data.edge_index, torch.ones(data.edge_index.shape[1]), torch.Size([data.x.shape[0], data.x.shape[0]])
        ).to_dense()
        recon_loss = F.binary_cross_entropy(mat, adj, reduction="sum")
        var = torch.exp(log_var)
        kl_loss = 0.5 * torch.mean(torch.sum(mean * mean + var - log_var - 1, dim = 1))
        print("recon_loss = %.3f, kl_loss = %.3f" % (recon_loss, kl_loss))
        return recon_loss + kl_loss

    def get_trainer(self, task, args):
        return GAETrainer
