import importlib

from .base_model import BaseModel


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
    "transe": "cogdl.models.emb.transe.TransE",
    "complex": "cogdl.models.emb.complex.ComplEx",
    "distmult": "cogdl.models.emb.distmult.DistMult",
    "rotate": "cogdl.models.emb.rotate.RotatE",
    "hope": "cogdl.models.emb.hope.HOPE",
    "spectral": "cogdl.models.emb.spectral.Spectral",
    "hin2vec": "cogdl.models.emb.hin2vec.Hin2vec",
    "netmf": "cogdl.models.emb.netmf.NetMF",
    "deepwalk": "cogdl.models.emb.deepwalk.DeepWalk",
    "gatne": "cogdl.models.emb.gatne.GATNE",
    "dgk": "cogdl.models.emb.dgk.DeepGraphKernel",
    "grarep": "cogdl.models.emb.grarep.GraRep",
    "dngr": "cogdl.models.emb.dngr.DNGR",
    "prone++": "cogdl.models.emb.pronepp.ProNEPP",
    "graph2vec": "cogdl.models.emb.graph2vec.Graph2Vec",
    "metapath2vec": "cogdl.models.emb.metapath2vec.Metapath2vec",
    "node2vec": "cogdl.models.emb.node2vec.Node2vec",
    "pte": "cogdl.models.emb.pte.PTE",
    "netsmf": "cogdl.models.emb.netsmf.NetSMF",
    "line": "cogdl.models.emb.line.LINE",
    "sdne": "cogdl.models.emb.sdne.SDNE",
    "prone": "cogdl.models.emb.prone.ProNE",
    "daegc": "cogdl.models.nn.daegc.DAEGC",
    "agc": "cogdl.models.nn.agc.AGC",
    "gae": "cogdl.models.nn.gae.GAE",
    "vgae": "cogdl.models.nn.gae.VGAE",
    "dgi": "cogdl.models.nn.dgi.DGIModel",
    "mvgrl": "cogdl.models.nn.mvgrl.MVGRL",
    "patchy_san": "cogdl.models.nn.patchy_san.PatchySAN",
    "gcn": "cogdl.models.nn.gcn.GCN",
    "actgcn": "cogdl.models.nn.actgcn.ActGCN",
    "gdc_gcn": "cogdl.models.nn.gdc_gcn.GDC_GCN",
    "graphsage": "cogdl.models.nn.graphsage.Graphsage",
    "compgcn": "cogdl.models.nn.compgcn.LinkPredictCompGCN",
    "drgcn": "cogdl.models.nn.drgcn.DrGCN",
    "unet": "cogdl.models.nn.graph_unet.GraphUnet",
    "gcnmix": "cogdl.models.nn.gcnmix.GCNMix",
    "diffpool": "cogdl.models.nn.diffpool.DiffPool",
    "gcnii": "cogdl.models.nn.gcnii.GCNII",
    "sign": "cogdl.models.nn.sign.SIGN",
    "mixhop": "cogdl.models.nn.mixhop.MixHop",
    "gat": "cogdl.models.nn.gat.GAT",
    "han": "cogdl.models.nn.han.HAN",
    "ppnp": "cogdl.models.nn.ppnp.PPNP",
    "grace": "cogdl.models.nn.grace.GRACE",
    "pprgo": "cogdl.models.nn.pprgo.PPRGo",
    "gin": "cogdl.models.nn.gin.GIN",
    "grand": "cogdl.models.nn.grand.Grand",
    "gtn": "cogdl.models.nn.gtn.GTN",
    "rgcn": "cogdl.models.nn.rgcn.LinkPredictRGCN",
    "deepergcn": "cogdl.models.nn.deepergcn.DeeperGCN",
    "drgat": "cogdl.models.nn.drgat.DrGAT",
    "infograph": "cogdl.models.nn.infograph.InfoGraph",
    "dropedge_gcn": "cogdl.models.nn.dropedge_gcn.DropEdge_GCN",
    "disengcn": "cogdl.models.nn.disengcn.DisenGCN",
    "mlp": "cogdl.models.nn.mlp.MLP",
    "sgc": "cogdl.models.nn.sgc.sgc",
    "sortpool": "cogdl.models.nn.sortpool.SortPool",
    "srgcn": "cogdl.models.nn.srgcn.SRGCN",
    "gcc": "cogdl.models.nn.gcc_model.GCCModel",
    "unsup_graphsage": "cogdl.models.nn.graphsage.UnsupGraphsage",
    "graphsaint": "cogdl.models.nn.graphsaint.GraphSAINT",
    "m3s": "cogdl.models.nn.m3s.M3S",
    "moe_gcn": "cogdl.models.nn.moe_gcn.MoEGCN",
    "lightgcn": "cogdl.models.nn.lightgcn.LightGCN",
    "correct_smooth_mlp": "cogdl.models.nn.correct_smooth.CorrectSmoothMLP",
    "sagn": "cogdl.models.nn.sagn.SAGN",
    "revgcn": "cogdl.models.nn.revgcn.RevGCN",
    "revgat": "cogdl.models.nn.revgcn.RevGAT",
    "revgen": "cogdl.models.nn.revgcn.RevGEN",
    "sage": "cogdl.models.nn.graphsage.SAGE",
    "autognn": "cogdl.models.nn.autognn.AutoGNN",
    "stgcn": "cogdl.models.nn.stgcn.STGCN",
    "stgat": "cogdl.models.nn.stgat.STGAT",
}
