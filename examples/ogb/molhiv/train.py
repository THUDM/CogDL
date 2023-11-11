import argparse
from cogdl import experiment
from cogdl.options import get_parser

from cogdl.datasets.ogb import OGBMolhivDataset

from modelwrapper import GraphClassificationModelWrapper
from datawrapper import GraphClassificationDataWrapper

from gnn import GNN

parser = get_parser()

parser.add_argument('--dataset', type=str, default="ogbg-molhiv", help='dataset name (default: ogbg-molhiv)')
parser.add_argument('--gnn', type=str, default='gin-virtual', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')

parser.add_argument("--scheduler-type", type=str, default=None, choices=[None, 'StepLR', 'PolynomialDecayLR'])
parser.add_argument("--scheduler-round", type=str, default='epoch', choices=['epoch', 'iteration'])
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument('--end_lr', type=float, default=1e-9)
parser.add_argument("--weight-decay", type=float, default=0.0)

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--warmup_epochs', type=int, default=6, help='number of epochs to warmup (default: 6)')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers (default: 4)')
parser.add_argument("--progress-bar", type=str, default='iteration', choices=['epoch', 'iteration'])

args = parser.parse_args()

args.mw = GraphClassificationModelWrapper
args.dw = GraphClassificationDataWrapper

dataset = OGBMolhivDataset()

args.metric_name = dataset.get_metric_name()

if args.gnn == 'gin':
    model = GNN(gnn_type = 'gin', num_tasks = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, dataset_name=args.dataset)
elif args.gnn == 'gin-virtual':
    model = GNN(gnn_type = 'gin', num_tasks = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, dataset_name=args.dataset)
elif args.gnn == 'gcn':
    model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, dataset_name=args.dataset)
elif args.gnn == 'gcn-virtual':
    model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, dataset_name=args.dataset)
else:
    raise ValueError('Invalid GNN type')

dataset_name = args.dataset

experiment(
    dataset = dataset,
    model = model,
    args = args,
)

"""
Result:
| Variant                   | test__metric   | val__metric   |
|---------------------------|----------------|---------------|
| (OGBMolhivDataset, 'GNN') | 0.7706±0.0000  | 0.8409±0.0000 |
"""