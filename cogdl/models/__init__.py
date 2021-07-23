import importlib

from .base_model import BaseModel
from cogdl.utils import init_operator_configs


init_operator_configs()

MODEL_REGISTRY = {}


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
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError("Model ({}: {}) must extend BaseModel".format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_model_cls


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            print(f"Failed to import {model} model.")
            return False
    return True


def build_model(args):
    if not try_import_model(args.model):
        exit(1)
    return MODEL_REGISTRY[args.model].build_model_from_args(args)


SUPPORTED_MODELS = {
    "hope": "cogdl.models.emb.hope",
    "spectral": "cogdl.models.emb.spectral",
    "hin2vec": "cogdl.models.emb.hin2vec",
    "netmf": "cogdl.models.emb.netmf",
    "distmult": "cogdl.models.emb.distmult",
    "transe": "cogdl.models.emb.transe",
    "deepwalk": "cogdl.models.emb.deepwalk",
    "rotate": "cogdl.models.emb.rotate",
    "gatne": "cogdl.models.emb.gatne",
    "dgk": "cogdl.models.emb.dgk",
    "grarep": "cogdl.models.emb.grarep",
    "dngr": "cogdl.models.emb.dngr",
    "prone++": "cogdl.models.emb.pronepp",
    "graph2vec": "cogdl.models.emb.graph2vec",
    "metapath2vec": "cogdl.models.emb.metapath2vec",
    "node2vec": "cogdl.models.emb.node2vec",
    "complex": "cogdl.models.emb.complex",
    "pte": "cogdl.models.emb.pte",
    "netsmf": "cogdl.models.emb.netsmf",
    "line": "cogdl.models.emb.line",
    "sdne": "cogdl.models.emb.sdne",
    "prone": "cogdl.models.emb.prone",
    "daegc": "cogdl.models.nn.daegc",
    "agc": "cogdl.models.nn.agc",
    "gae": "cogdl.models.nn.gae",
    "vgae": "cogdl.models.nn.gae",
    "dgi": "cogdl.models.nn.dgi",
    "dgi_sampling": "cogdl.models.nn.dgi",
    "mvgrl": "cogdl.models.nn.mvgrl",
    "patchy_san": "cogdl.models.nn.patchy_san",
    "chebyshev": "cogdl.models.nn.pyg_cheb",
    "gcn": "cogdl.models.nn.gcn",
    "gdc_gcn": "cogdl.models.nn.gdc_gcn",
    "hgpsl": "cogdl.models.nn.pyg_hgpsl",
    "graphsage": "cogdl.models.nn.graphsage",
    "compgcn": "cogdl.models.nn.compgcn",
    "drgcn": "cogdl.models.nn.drgcn",
    "gpt_gnn": "cogdl.models.nn.pyg_gpt_gnn",
    "unet": "cogdl.models.nn.pyg_graph_unet",
    "gcnmix": "cogdl.models.nn.gcnmix",
    "diffpool": "cogdl.models.nn.diffpool",
    "gcnii": "cogdl.models.nn.gcnii",
    "sign": "cogdl.models.nn.sign",
    "pyg_gcn": "cogdl.models.nn.pyg_gcn",
    "mixhop": "cogdl.models.nn.mixhop",
    "gat": "cogdl.models.nn.gat",
    "han": "cogdl.models.nn.han",
    "ppnp": "cogdl.models.nn.ppnp",
    "grace": "cogdl.models.nn.grace",
    "jknet": "cogdl.models.nn.dgl_jknet",
    "pprgo": "cogdl.models.nn.pprgo",
    "gin": "cogdl.models.nn.gin",
    "dgcnn": "cogdl.models.nn.pyg_dgcnn",
    "grand": "cogdl.models.nn.grand",
    "gtn": "cogdl.models.nn.pyg_gtn",
    "rgcn": "cogdl.models.nn.rgcn",
    "deepergcn": "cogdl.models.nn.deepergcn",
    "drgat": "cogdl.models.nn.drgat",
    "infograph": "cogdl.models.nn.infograph",
    "dropedge_gcn": "cogdl.models.nn.dropedge_gcn",
    "disengcn": "cogdl.models.nn.disengcn",
    "fastgcn": "cogdl.models.nn.fastgcn",
    "mlp": "cogdl.models.nn.mlp",
    "sgc": "cogdl.models.nn.sgc",
    "stpgnn": "cogdl.models.nn.stpgnn",
    "sortpool": "cogdl.models.nn.sortpool",
    "srgcn": "cogdl.models.nn.pyg_srgcn",
    "asgcn": "cogdl.models.nn.asgcn",
    "gcc": "cogdl.models.nn.dgl_gcc",
    "unsup_graphsage": "cogdl.models.nn.unsup_graphsage",
    "sagpool": "cogdl.models.nn.pyg_sagpool",
    "graphsaint": "cogdl.models.nn.graphsaint",
    "m3s": "cogdl.models.nn.m3s",
    "supergat": "cogdl.models.nn.pyg_supergat",
    "self_auxiliary_task": "cogdl.models.nn.self_auxiliary_task",
    "moe_gcn": "cogdl.models.nn.moe_gcn",
    "lightgcn": "cogdl.models.nn.lightgcn",
    "correct_smooth": "cogdl.models.nn.correct_smooth",
    "correct_smooth_mlp": "cogdl.models.nn.correct_smooth",
    "sagn": "cogdl.models.nn.sagn",
    "revgcn": "cogdl.models.nn.revgcn",
    "revgat": "cogdl.models.nn.revgcn",
    "revgen": "cogdl.models.nn.revgcn",
    "sage": "cogdl.models.nn.graphsage",
}
