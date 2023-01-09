import torch
import torch.nn.functional as F

from cogdl.options import get_default_args
from cogdl.experiments import train


cuda_available = torch.cuda.is_available()
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


def test_gdc_gcn_cora():
    args = get_default_args_for_nc("cora", "gdc_gcn")
    args.num_layers = 1
    args.alpha = 0.05  # ppr filter param
    args.t = 5.0  # heat filter param
    args.k = 128  # top k entries to be retained
    args.eps = 0.01  # change depending on gdc_type
    for gdc_type in ["none", "ppr", "heat"]:
        args.gdc_type = gdc_type
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1


def test_gcn_cora():
    args = get_default_args_for_nc("cora", "gcn")
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


def test_gat_cora():
    args = get_default_args_for_nc("cora", "gat")
    args.alpha = 0.2
    args.attn_drop = 0.2
    args.nhead = 8
    args.last_nhead = 2
    args.num_layers = 3
    args.residual = True
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_mlp_pubmed():
    args = get_default_args_for_nc("pubmed", "mlp")
    args.num_layers = 2
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_mixhop_citeseer():
    args = get_default_args_for_nc("citeseer", "mixhop")
    args.layer1_pows = [20, 20, 20]
    args.layer2_pows = [20, 20, 20]
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_graphsage_cora():
    args = get_default_args_for_nc("cora", "graphsage", dw="graphsage_dw", mw="graphsage_mw")
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


def test_clustergcn_pubmed():
    args = get_default_args_for_nc("pubmed", "gcn", dw="cluster_dw")
    args.cpu = True
    args.batch_size = 3
    args.n_cluster = 20
    args.eval_step = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_graphsaint_cora():
    args = get_default_args_for_nc("cora", "graphsaint")
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


def test_unet_citeseer():
    args = get_default_args_for_nc("citeseer", "unet")
    args.cpu = True
    args.pool_rate = [0.5, 0.5]
    args.n_pool = 2
    args.adj_dropout = 0.3
    args.n_dropout = 0.8
    args.hidden_size = 16
    args.improved = True
    args.aug_adj = True
    args.activation = "elu"
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_drgcn_cora():
    args = get_default_args_for_nc("cora", "drgcn")
    args.num_layers = 2
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_drgat_cora():
    args = get_default_args_for_nc("cora", "drgat")
    args.nhead = 8
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_disengcn_cora():
    args = get_default_args_for_nc("cora", "disengcn")
    args.K = [4, 2]
    args.activation = "leaky_relu"
    args.tau = 1.0
    args.iterations = 3
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_graph_mix():
    args = get_default_args_for_nc("cora", "gcnmix", mw="gcnmix_mw")
    args.epochs = 2
    args.rampup_starts = 1
    args.rampup_ends = 100
    args.mixup_consistency = 5.0
    args.ema_decay = 0.999
    args.alpha = 1.0
    args.temperature = 1.0
    args.k = 10
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_srgcn_cora():
    args = get_default_args_for_nc("cora", "srgcn")
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


def test_gcnii_cora():
    args = get_default_args_for_nc("cora", "gcnii")
    args.dataset = "cora"
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


def test_deepergcn_cora():
    args = get_default_args_for_nc("cora", "deepergcn")
    args.n_cluster = 10
    args.num_layers = 2
    args.connection = "res+"
    args.cluster_number = 3
    args.epochs = 2
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
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_grand_cora():
    args = get_default_args_for_nc("cora", "grand", mw="grand_mw")
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


def test_sign_cora():
    args = get_default_args_for_nc("cora", "sign")
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


def test_ppnp_cora():
    args = get_default_args_for_nc("cora", "ppnp")
    args.num_layers = 2
    args.propagation_type = "ppnp"
    args.alpha = 0.1
    args.num_iterations = 10

    ret = train(args)
    assert 0 < ret["test_acc"] < 1


def test_appnp_cora():
    args = get_default_args_for_nc("cora", "ppnp")
    args.num_layers = 2
    args.propagation_type = "appnp"
    args.alpha = 0.1
    args.num_iterations = 10

    ret = train(args)
    assert 0 < ret["test_acc"] < 1


def test_sgc_cora():
    args = get_default_args_for_nc("cora", "sgc")

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_dropedge_gcn_cora():
    args = get_default_args_for_nc("cora", "dropedge_gcn")
    args.baseblock = "mutigcn"
    args.inputlayer = "gcn"
    args.outputlayer = "gcn"
    args.hidden_size = 64
    args.dropout = 0.5
    args.withbn = False
    args.withloop = False
    args.nhiddenlayer = 1
    args.nbaseblocklayer = 1
    args.aggrmethod = "default"
    args.activation = F.relu
    args.task_type = "full"

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_dropedge_resgcn_cora():
    args = get_default_args_for_nc("cora", "dropedge_gcn")
    args.baseblock = "resgcn"
    args.inputlayer = "gcn"
    args.outputlayer = "gcn"
    args.hidden_size = 64
    args.dropout = 0.5
    args.withbn = False
    args.withloop = False
    args.nhiddenlayer = 1
    args.nbaseblocklayer = 1
    args.aggrmethod = "concat"
    args.activation = F.relu
    args.task_type = "full"

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_dropedge_densegcn_cora():
    args = get_default_args_for_nc("cora", "dropedge_gcn")
    args.baseblock = "densegcn"
    args.inputlayer = ""
    args.outputlayer = "none"
    args.hidden_size = 64
    args.dropout = 0.5
    args.withbn = False
    args.withloop = False
    args.nhiddenlayer = 1
    args.nbaseblocklayer = 1
    args.aggrmethod = "add"
    args.activation = F.relu
    args.task_type = "full"

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_dropedge_inceptiongcn_cora():
    args = get_default_args_for_nc("cora", "dropedge_gcn")

    args.baseblock = "inceptiongcn"
    args.inputlayer = "gcn"
    args.outputlayer = "gcn"
    args.hidden_size = 64
    args.dropout = 0.5
    args.withbn = False
    args.withloop = False
    args.nhiddenlayer = 1
    args.nbaseblocklayer = 1
    args.aggrmethod = "add"
    args.activation = F.relu
    args.task_type = "full"

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_pprgo_cora():
    args = get_default_args_for_nc("cora", "pprgo", dw="pprgo_dw", mw="pprgo_mw")
    args.cpu = True
    args.k = 32
    args.alpha = 0.5
    args.eval_step = 1
    args.batch_size = 32
    args.test_batch_size = 128
    args.activation = "relu"
    args.num_layers = 2
    args.nprop_inference = 2
    args.eps = 0.001
    for norm in ["sym", "row"]:
        args.norm = norm
        ret = train(args)
        assert 0 <= ret["test_acc"] <= 1

    args.test_batch_size = 0
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_gcn_ppi():
    args = get_default_args_for_nc("ppi", "gcn")
    args.cpu = True

    ret = train(args)
    assert 0 <= ret["test_micro_f1"] <= 1


def test_sagn_cora():
    args = get_default_args_for_nc("cora", "sagn", dw="sagn_dw", mw="sagn_mw")
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


def test_c_s_cora():
    args = get_default_args_for_nc("cora", "correct_smooth_mlp")
    args.use_embeddings = True
    args.correct_alpha = 0.5
    args.smooth_alpha = 0.5
    args.num_correct_prop = 2
    args.num_smooth_prop = 2
    args.correct_norm = "sym"
    args.smooth_norm = "sym"
    args.scale = 1.0
    args.autoscale = True

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1

    args.autoscale = False
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_sage_cora():
    args = get_default_args_for_nc("cora", "sage")
    args.aggr = "mean"
    args.normalize = True
    args.norm = "layernorm"

    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_revnets_cora():
    args = get_default_args_for_nc("cora", "revgen")
    args.group = 2
    args.num_layers = 3
    args.last_norm = "batchnorm"
    args.p = args.beta = 1.0
    args.learn_p = args.learn_beta = True
    args.use_msg_norm = False
    args.learn_msg_scale = False
    args.aggr = "mean"
    assert 0 <= train(args)["test_acc"] <= 1

    args.model = "revgat"
    args.nhead = 2
    args.alpha = 0.2
    args.norm = "batchnorm"
    args.residual = True
    args.attn_drop = 0.2
    args.last_nhead = 1
    args.drop_edge_rate = 0.0
    assert 0 <= train(args)["test_acc"] <= 1

    args.model = "revgcn"
    assert 0 <= train(args)["test_acc"] <= 1


def test_gcc_cora():
    args = get_default_args_for_nc("cora", "gcc", mw="gcc_mw", dw="gcc_dw")
    args.pretrain = True
    args.unsup = True
    args.parallel = False
    args.epochs = 1
    args.num_workers = 1
    args.num_copies = 1
    args.batch_size = 16
    args.rw_hops = 8
    args.subgraph_size = 16
    args.positional_embedding_size = 16
    args.nce_k = 4
    train(args)


if __name__ == "__main__":
    test_gdc_gcn_cora()
    test_gcn_cora()
    test_gat_cora()
    test_sgc_cora()
    test_mlp_pubmed()
    test_mixhop_citeseer()
    test_graphsage_cora()
    test_disengcn_cora()
    test_graph_mix()
    test_srgcn_cora()
    test_gcnii_cora()
    test_deepergcn_cora()
    test_grand_cora()
    test_graphsaint_cora()
    test_sign_cora()
    test_ppnp_cora()
    test_appnp_cora()
    test_dropedge_gcn_cora()
    test_dropedge_resgcn_cora()
    test_dropedge_inceptiongcn_cora()
    test_dropedge_densegcn_cora()
    test_revnets_cora()
    test_gcn_ppi()
    test_gcc_cora()
    test_pprgo_cora()
    test_sagn_cora()
    test_clustergcn_pubmed()
