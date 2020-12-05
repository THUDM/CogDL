import torch
from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "hidden_size": 16,
        "dropout": 0.5,
        "patience": 2,
        "device_id": [0],
        "max_epoch": 3,
        "sampler": "none",
        "cpu": not cuda_available,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "missing_rate": -1,
    }
    return build_args_from_dict(default_dict)


def test_gdc_gcn_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'gdc_gcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 1
    args.alpha = 0.05 # ppr filter param
    args.t = 5.0 # heat filter param
    args.k = 128 # top k entries to be retained
    args.eps = 0.01 # change depending on gdc_type
    args.dataset = dataset
    args.gdc_type = 'ppr' # ppr, heat, none

    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1

def test_gcn_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "gcn"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_gat_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "gat"
    args.alpha = 0.2
    args.nheads = 8
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_mlp_pubmed():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "pubmed"
    args.model = "mlp"
    args.num_layers = 2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_mixhop_citeseer():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "citeseer"
    args.model = "mixhop"
    args.layer1_pows = [20, 20, 20]
    args.layer2_pows = [20, 20, 20]
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pairnorm_cora_deepgcn():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pairnorm"
    args.pn_model = "DeepGCN"
    args.nlayer = 10
    args.missing_rate = 100
    args.norm_mode = 'PN-SI'
    args.residual = 0
    args.hidden_layers = 64
    args.nhead = 1
    args.dropout = 0.6
    args.norm_scale = 1.0
    args.no_fea_norm = "store_false"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <=1


def test_pairnorm_cora_gcn():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pairnorm"
    args.pn_model = "GCN"
    args.nlayer = 10
    args.missing_rate = 100
    args.norm_mode = 'PN-SI'
    args.residual = 0
    args.hidden_layers = 64
    args.nhead = 1
    args.dropout = 0.6
    args.norm_scale = 1.0
    args.no_fea_norm = "store_false"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <=1


def test_pairnorm_cora_sgc():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pairnorm"
    args.pn_model = "SGC"
    args.nlayer = 10
    args.missing_rate = 100
    args.norm_mode = 'PN-SI'
    args.residual = 0
    args.hidden_layers = 64
    args.nhead = 1
    args.dropout = 0.6
    args.norm_scale = 1.0
    args.no_fea_norm = "store_false"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <=1


def test_pairnorm_cora_deepgat():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pairnorm"
    args.pn_model = "DeepGAT"
    args.nlayer = 10
    args.missing_rate = 100
    args.norm_mode = 'PN-SI'
    args.residual = 0
    args.hidden_layers = 64
    args.nhead = 1
    args.dropout = 0.6
    args.norm_scale = 1.0
    args.no_fea_norm = "store_false"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <=1


def test_graphsage_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "graphsage"
    args.num_layers = 2
    args.hidden_size = [128]
    args.sample_size = [10, 10]
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_cheb_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "chebyshev"
    args.num_layers = 2
    args.filter_size = 5
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_gcn_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pyg_gcn"
    args.num_layers = 2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_gcn_cora_sampler():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pyg_gcn"
    args.cpu = True
    args.num_layers = 2
    args.sample_coverage = 20
    args.size_subgraph = 200
    args.num_walks = 20
    args.walk_length = 10
    args.size_frontier = 20
    sampler_list = ["node", "edge", "rw", "mrw"]

    for sampler in sampler_list:
        args.sampler = sampler
        task = build_task(args)
        ret = task.train()
        assert 0 <= ret["Acc"] <= 1


def test_pyg_gat_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "pyg_gat"
    args.num_heads = 8
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_infomax_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "infomax"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_unet_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "unet"
    args.num_layers = 2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_drgcn_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "drgcn"
    args.num_layers = 2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pyg_drgat_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "drgat"
    args.num_heads = 8
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_disengcn_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "disengcn"
    args.K = [4, 2]
    args.activation = "leaky_relu"
    args.tau = 1.0
    args.iterations = 3
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_graph_mix():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "gcnmix"
    args.rampup_starts = 1
    args.rampup_ends = 100
    args.mixup_consistency = 5.0
    args.ema_decay = 0.999
    args.alpha = 1.0
    args.temperature = 1.0
    args.k = 10
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_srgcn_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "srgcn"
    args.num_heads = 4
    args.subheads = 1
    args.nhop = 1
    args.node_dropout = 0.5
    args.alpha = 0.2

    args.normalization = "identity"
    args.attention_type = "identity"
    args.activation = "linear"

    norm_list = ["identity", "row_uniform", "row_softmax", "col_uniform", "symmetry"]
    activation_list = ["relu", "relu6", "sigmoid", "tanh", "leaky_relu", "softplus", "elu", "linear"]
    attn_list = ["node", "edge", "identity", "heat", "ppr"]  # gaussian

    for norm in norm_list:
        args.normalization = norm
        task = build_task(args)
        ret = task.train()
        assert 0 < ret["Acc"] < 1

    args.norm = "identity"
    for ac in activation_list:
        args.activation = ac
        task = build_task(args)
        ret = task.train()
        assert 0 < ret["Acc"] < 1

    args.activation = "relu"
    for attn in attn_list:
        args.attention_type = attn
        task = build_task(args)
        ret = task.train()
        assert 0 < ret["Acc"] < 1


def test_gcnii_cora():
    args = get_default_args()
    args.dataset = "cora"
    args.task = "node_classification"
    args.model = "gcnii"
    args.num_layers = 2
    args.lmbda = 0.2
    args.wd1 = 0.001
    args.wd2 = 5e-4
    args.alpha = 0.1
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] < 1


def test_deepergcn_cora():
    args = get_default_args()
    args.dataset = "cora"
    args.task = "node_classification"
    args.model = "deepergcn"
    args.num_layers = 2
    args.connection = "res+"
    args.cluster_number = 3
    args.max_epoch = 10
    args.patience = 1
    args.learn_beta = True
    args.learn_msg_scale = True
    args.aggr = "softmax_sg"
    args.batch_size = 1
    args.activation = "relu"
    args.beta = 1.0
    args.p = 1.0
    args.use_msg_norm = True
    args.learn_p = True
    args.learn_beta = True
    args.learn_msg_scale = True
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] < 1


def test_grand_cora():
    args = get_default_args()
    args.model = "grand"
    args.dataset = "cora"
    args.task = "node_classification"
    args.hidden_dropout = 0.5
    args.order = 4
    args.input_dropout = 0.5
    args.lam = 1
    args.tem = 0.3
    args.sample = 2
    args.alpha = 0.1
    args.dropnode_rate = 0.5
    args.bn = True
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] < 1


def test_gpt_gnn_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "gpt_gnn"
    args.use_pretrain = False
    args.pretrain_model_dir = ""
    args.task_name = ""
    args.sample_depth = 3
    args.sample_width = 16
    args.conv_name = "hgt"
    args.n_hid = 16
    args.n_heads = 2
    args.n_layers = 2
    args.prev_norm = True
    args.last_norm = True
    args.optimizer = "adamw"
    args.scheduler = "cosine"
    args.data_percentage = 0.1
    args.n_epoch = 2
    args.n_pool = 8
    args.n_batch = 5
    args.batch_size = 64
    args.clip = 0.5
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1

def test_jknet_jknet_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "jknet_cora"
    args.model = "jknet"
    args.lr = 0.005
    args.layer_aggregation = 'maxpool'
    args.node_aggregation = 'sum'
    args.n_layers = 6
    args.n_units = 16
    args.in_features = 1433
    args.out_features = 7
    args.max_epoch = 100
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1

def test_ppnp_citeseer():
    args = get_default_args()
    args.task = 'node_classification'
    args.model = 'ppnp'
    args.dataset = 'citeseer'
    args.propagation_type = 'ppnp'
    args.alpha = 0.1
    args.num_iterations = 10
    task = build_task(args)
    ret = task.train()
    assert 0 < ret['Acc'] < 1

def test_appnp_citeseer():
    args = get_default_args()
    args.task = 'node_classification'
    args.model = 'ppnp'
    args.dataset = 'citeseer'
    args.propagation_type = 'appnp'
    args.alpha = 0.1
    args.num_iterations = 10
    task = build_task(args)
    ret = task.train()
    assert 0 < ret['Acc'] < 1

def test_sgcpn_cora():
    args = get_default_args()
    args.dataset = "cora"
    args.task = "node_classification"
    args.model = "sgcpn"
    args.dropout = 0.6
    args.num_layers = 10
    args.norm_mode = "PN"
    args.norm_scale = 10
    args.missing_rate = 20
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] < 1

def test_sgc_cora():
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "sgc"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1

if __name__ == "__main__":
    test_gdc_gcn_cora()
    test_gcn_cora()
    test_gat_cora()
    test_pairnorm_cora_deepgcn()
    test_pairnorm_cora_deepgat()
    test_pairnorm_cora_gcn()
    test_pairnorm_cora_sgc()
    test_sgcpn_cora()
    test_sgc_cora()
    test_mlp_pubmed()
    test_mixhop_citeseer()
    test_graphsage_cora()
    test_pyg_cheb_cora()
    test_pyg_gcn_cora()
    test_pyg_gat_cora()
    test_pyg_infomax_cora()
    test_pyg_unet_cora()
    test_pyg_drgcn_cora()
    test_pyg_drgat_cora()
    test_disengcn_cora()
    test_graph_mix()
    test_srgcn_cora()
    test_gcnii_cora()
    test_deepergcn_cora()
    test_grand_cora()
    test_pyg_gcn_cora_sampler()
    test_gpt_gnn_cora()
    test_jknet_jknet_cora()
    test_ppnp_citeseer()
    test_appnp_citeseer()
    test_sgcpn_cora()
