import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from .. import BaseModel, register_model

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

@register_model("infomax")
class Infomax(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=512)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
        )

    def __init__(self, num_features, num_classes, hidden_size):
        super(Infomax, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        self.model = DeepGraphInfomax(
            hidden_channels=hidden_size, encoder=Encoder(num_features, hidden_size),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def loss(self, data):
        pos_z, neg_z, summary = self.forward(data.x, data.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)
        return loss
    
    def predict(self, data):
        z, _, _ = self.forward(data.x, data.edge_index)
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150)
        clf.fit(z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
        logits = torch.Tensor(clf.predict_proba(z.detach().cpu().numpy()))
        if z.is_cuda:
            logits = logits.cuda()
        return logits
