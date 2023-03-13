import importlib
from cogdl.backend import BACKEND

if BACKEND == "jittor":
    from .jittor import *
    from .jittor.base_model import BaseModel
elif BACKEND == "torch":
    from .torch import *
    from .torch.base_model import BaseModel
else:
    raise ("Unsupported backend:", BACKEND)


def register_model(name):
    """
    New model types can be added to cogdl with the :func:`register_model`
    function decorator.
    For example::
        @register_model('gat')
        class GAT(BaseModel):
            (...)
    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        print("The `register_model` API is deprecated!")
        return cls

    return register_model_cls


def try_adding_model_args(model, parser):
    if model in SUPPORTED_MODELS:
        path = ".".join(SUPPORTED_MODELS[model].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = SUPPORTED_MODELS[model].split(".")[-1]
        getattr(module, class_name).add_args(parser)


def build_model(args):
    model = args.model
    if isinstance(model, list):
        model = model[0]
    if model in SUPPORTED_MODELS:
        path = ".".join(SUPPORTED_MODELS[model].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        raise NotImplementedError(f"Failed to import {model} model.")
    class_name = SUPPORTED_MODELS[model].split(".")[-1]
    return getattr(module, class_name).build_model_from_args(args)


SUPPORTED_MODELS = {
    # "gcn": f"cogdl.models.{BACKEND}.nn.gcn.GCN",
    # "gat": f"cogdl.models.{BACKEND}.nn.gat.GAT",
    # "grand": f"cogdl.models.{BACKEND}.nn.grand.Grand",
    # "gcnii": f"cogdl.models.{BACKEND}.nn.gcnii.GCNII",
    # "dgi": f"cogdl.models.{BACKEND}.nn.dgi.DGIModel",
    # "graphsage": f"cogdl.models.{BACKEND}.nn.graphsage.Graphsage",
    # "drgat": f"cogdl.models.{BACKEND}.nn.drgat.DrGAT",
    # "mvgrl": f"cogdl.models.{BACKEND}.nn.mvgrl.MVGRL",
    "transe": f"cogdl.models.{BACKEND}.emb.transe.TransE",
    "complex": f"cogdl.models.{BACKEND}.emb.complex.ComplEx",
    "distmult": f"cogdl.models.{BACKEND}.emb.distmult.DistMult",
    "rotate": f"cogdl.models.{BACKEND}.emb.rotate.RotatE",
    "hope": f"cogdl.models.{BACKEND}.emb.hope.HOPE",
    "spectral": f"cogdl.models.{BACKEND}.emb.spectral.Spectral",
    "hin2vec": f"cogdl.models.{BACKEND}.emb.hin2vec.Hin2vec",
    "netmf": f"cogdl.models.{BACKEND}.emb.netmf.NetMF",
    "deepwalk": f"cogdl.models.{BACKEND}.emb.deepwalk.DeepWalk",
    "gatne": f"cogdl.models.{BACKEND}.emb.gatne.GATNE",
    "dgk": f"cogdl.models.{BACKEND}.emb.dgk.DeepGraphKernel",
    "grarep": f"cogdl.models.{BACKEND}.emb.grarep.GraRep",
    "dngr": f"cogdl.models.{BACKEND}.emb.dngr.DNGR",
    "prone++": f"cogdl.models.{BACKEND}.emb.pronepp.ProNEPP",
    "graph2vec": f"cogdl.models.{BACKEND}.emb.graph2vec.Graph2Vec",
    "metapath2vec": f"cogdl.models.{BACKEND}.emb.metapath2vec.Metapath2vec",
    "node2vec": f"cogdl.models.{BACKEND}.emb.node2vec.Node2vec",
    "pte": f"cogdl.models.{BACKEND}.emb.pte.PTE",
    "netsmf": f"cogdl.models.{BACKEND}.emb.netsmf.NetSMF",
    "line": f"cogdl.models.{BACKEND}.emb.line.LINE",
    "sdne": f"cogdl.models.{BACKEND}.emb.sdne.SDNE",
    "prone": f"cogdl.models.{BACKEND}.emb.prone.ProNE",
    "daegc": f"cogdl.models.{BACKEND}.nn.daegc.DAEGC",
    "agc": f"cogdl.models.{BACKEND}.nn.agc.AGC",
    "gae": f"cogdl.models.{BACKEND}.nn.gae.GAE",
    "vgae": f"cogdl.models.{BACKEND}.nn.gae.VGAE",
    "dgi": f"cogdl.models.{BACKEND}.nn.dgi.DGIModel",
    "mvgrl": f"cogdl.models.{BACKEND}.nn.mvgrl.MVGRL",
    "patchy_san": f"cogdl.models.{BACKEND}.nn.patchy_san.PatchySAN",
    "gcn": f"cogdl.models.{BACKEND}.nn.gcn.GCN",
    "actgcn": f"cogdl.models.{BACKEND}.nn.actgcn.ActGCN",
    "gdc_gcn": f"cogdl.models.{BACKEND}.nn.gdc_gcn.GDC_GCN",
    "graphsage": f"cogdl.models.{BACKEND}.nn.graphsage.Graphsage",
    "compgcn": f"cogdl.models.{BACKEND}.nn.compgcn.LinkPredictCompGCN",
    "drgcn": f"cogdl.models.{BACKEND}.nn.drgcn.DrGCN",
    "unet": f"cogdl.models.{BACKEND}.nn.graph_unet.GraphUnet",
    "gcnmix": f"cogdl.models.{BACKEND}.nn.gcnmix.GCNMix",
    "diffpool": f"cogdl.models.{BACKEND}.nn.diffpool.DiffPool",
    "gcnii": f"cogdl.models.{BACKEND}.nn.gcnii.GCNII",
    "sign": f"cogdl.models.{BACKEND}.nn.sign.SIGN",
    "mixhop": f"cogdl.models.{BACKEND}.nn.mixhop.MixHop",
    "gat": f"cogdl.models.{BACKEND}.nn.gat.GAT",
    "han": f"cogdl.models.{BACKEND}.nn.han.HAN",
    "ppnp": f"cogdl.models.{BACKEND}.nn.ppnp.PPNP",
    "grace": f"cogdl.models.{BACKEND}.nn.grace.GRACE",
    "pprgo": f"cogdl.models.{BACKEND}.nn.pprgo.PPRGo",
    "gin": f"cogdl.models.{BACKEND}.nn.gin.GIN",
    "grand": f"cogdl.models.{BACKEND}.nn.grand.Grand",
    "gtn": f"cogdl.models.{BACKEND}.nn.gtn.GTN",
    "rgcn": f"cogdl.models.{BACKEND}.nn.rgcn.LinkPredictRGCN",
    "deepergcn": f"cogdl.models.{BACKEND}.nn.deepergcn.DeeperGCN",
    "drgat": f"cogdl.models.{BACKEND}.nn.drgat.DrGAT",
    "infograph": f"cogdl.models.{BACKEND}.nn.infograph.InfoGraph",
    "dropedge_gcn": f"cogdl.models.{BACKEND}.nn.dropedge_gcn.DropEdge_GCN",
    "disengcn": f"cogdl.models.{BACKEND}.nn.disengcn.DisenGCN",
    "mlp": f"cogdl.models.{BACKEND}.nn.mlp.MLP",
    "sgc": f"cogdl.models.{BACKEND}.nn.sgc.sgc",
    "sortpool": f"cogdl.models.{BACKEND}.nn.sortpool.SortPool",
    "srgcn": f"cogdl.models.{BACKEND}.nn.srgcn.SRGCN",
    "gcc": f"cogdl.models.{BACKEND}.nn.gcc_model.GCCModel",
    "unsup_graphsage": f"cogdl.models.{BACKEND}.nn.graphsage.Graphsage",
    "graphsaint": f"cogdl.models.{BACKEND}.nn.graphsaint.GraphSAINT",
    "m3s": f"cogdl.models.{BACKEND}.nn.m3s.M3S",
    "moe_gcn": f"cogdl.models.{BACKEND}.nn.moe_gcn.MoEGCN",
    "lightgcn": f"cogdl.models.{BACKEND}.nn.lightgcn.LightGCN",
    "correct_smooth_mlp": f"cogdl.models.{BACKEND}.nn.correct_smooth.CorrectSmoothMLP",
    "sagn": f"cogdl.models.{BACKEND}.nn.sagn.SAGN",
    "revgcn": f"cogdl.models.{BACKEND}.nn.revgcn.RevGCN",
    "revgat": f"cogdl.models.{BACKEND}.nn.revgcn.RevGAT",
    "revgen": f"cogdl.models.{BACKEND}.nn.revgcn.RevGEN",
    "sage": f"cogdl.models.{BACKEND}.nn.graphsage.SAGE",
    "autognn": f"cogdl.models.{BACKEND}.nn.autognn.AutoGNN",
    "stgcn": f"cogdl.models.{BACKEND}.nn.stgcn.STGCN",
    "stgat": f"cogdl.models.{BACKEND}.nn.stgat.STGAT",
}
