import os

os.environ["CogDLBACKEND"] = "jittor"
from cogdl import function as BF
from jittor import nn 

from cogdl.options import get_default_args
from cogdl.experiments_jt import train


cuda_available = BF.cuda_is_available()
default_dict = {
    "hidden_size": 16,
    "dropout": 0.5,
    "patience": 2,
    "epochs": 3,
    "sampler": "none",
    "num_layers": 2,
    "cpu": not cuda_available,
    "checkpoint": False,
    "auxiliary_task": "none",
    "eval_step": 1,
    "activation": "relu",
    "residual": False,
    "norm": None,
    "num_workers": 1,
}


def get_default_args_for_nc(dataset, model, dw="node_classification_dw", mw="node_classification_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args



def test_gcn_citeseer():
    args = get_default_args_for_nc("citeseer", "gcn")
    args.num_layers = 2
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1
    for n in ["batchnorm", "layernorm"]:
        args.norm = n
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1
    args.residual = True
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_gat_citeseer():
    args = get_default_args_for_nc("citeseer", "gat")
    args.alpha = 0.2
    args.attn_drop = 0.2
    args.nhead = 8
    args.last_nhead = 2
    args.num_layers = 3
    args.residual = True
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_mlp_citeseer():
    args = get_default_args_for_nc("citeseer", "mlp")
    args.num_layers = 2
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_mixhop_citeseer():
    args = get_default_args_for_nc("citeseer", "mixhop")
    args.layer1_pows = [20, 20, 20]
    args.layer2_pows = [20, 20, 20]
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_graphsage_citeseer():
    args = get_default_args_for_nc("citeseer", "graphsage", dw="graphsage_dw", mw="graphsage_mw")
    args.aggr = "mean"
    args.batch_size = 32
    args.num_layers = 2
    args.patience = 1
    args.epochs = 2
    args.hidden_size = [32, 32]
    args.sample_size = [3, 5]
    args.num_workers = 1
    args.eval_step = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_graphsaint_citeseer():
    args = get_default_args_for_nc("citeseer", "graphsaint")
    args.eval_cpu = True
    args.batch_size = 10
    args.cpu = True
    args.architecture = "1-1-0"
    args.aggr = "concat"
    args.act = "relu"
    args.bias = "norm"
    args.sample_coverage = 10
    args.size_subgraph = 200
    args.num_walks = 20
    args.walk_length = 10
    args.size_frontier = 20
    args.method = "node"
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_drgat_citeseer():
    args = get_default_args_for_nc("citeseer", "drgat")
    args.nhead = 8
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_srgcn_citeseer():
    args = get_default_args_for_nc("citeseer", "srgcn")
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
    attn_list = ["node", "edge", "identity", "heat", "ppr", "gaussian"]

    for norm in norm_list:
        args.normalization = norm
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1

    args.norm = "identity"
    for ac in activation_list:
        args.activation = ac
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1

    args.activation = "relu"
    for attn in attn_list:
        args.attention_type = attn
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1


def test_gcnii_citeseer():
    args = get_default_args_for_nc("citeseer", "gcnii")
    args.dataset = "citeseer"
    args.model = "gcnii"
    args.num_layers = 2
    args.lmbda = 0.2
    args.wd1 = 0.001
    args.wd2 = 5e-4
    args.alpha = 0.1
    for residual in [False, True]:
        args.residual = residual
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1


def test_grand_citeseer():
    args = get_default_args_for_nc("citeseer", "grand", mw="grand_mw")
    args.hidden_dropout = 0.5
    args.order = 4
    args.input_dropout = 0.5
    args.lam = 1
    args.tem = 0.3
    args.sample = 2
    args.alpha = 0.1
    args.dropnode_rate = 0.5
    args.bn = True
    ret = train(args)
    assert 0 < ret["test_acc"] < 1


def test_sign_citeseer():
    args = get_default_args_for_nc("citeseer", "sign")
    args.lr = 0.00005
    args.hidden_size = 2048
    args.num_layers = 3
    args.nhop = 3
    args.dropout = 0.3
    args.directed = False
    args.dropedge_rate = 0.2
    args.adj_norm = [
        "sym",
    ]
    args.remove_diag = False
    args.diffusion = "ppr"

    ret = train(args)
    assert 0 < ret["test_acc"] < 1
    args.diffusion = "sgc"
    ret = train(args)
    assert 0 < ret["test_acc"] < 1



def test_sagn_citeseer():
    args = get_default_args_for_nc("citeseer", "sagn", dw="sagn_dw", mw="sagn_mw")
    args.nhop = args.label_nhop = 2
    args.threshold = 0.5
    args.use_labels = True
    args.nstage = 2
    args.batch_size = 32
    args.data_gpu = False
    args.attn_drop = 0.0
    args.input_drop = 0.0
    args.nhead = 2
    args.negative_slope = 0.2
    args.mlp_layer = 2
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


if __name__ == "__main__":
    test_gcn_citeseer()
    test_mlp_citeseer()
    test_gat_citeseer()
    test_drgat_citeseer()
    test_graphsage_citeseer()
    test_gcnii_citeseer()
    test_grand_citeseer()

