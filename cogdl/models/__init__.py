from .base_model import BaseModel


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


def build_model(args):
    return MODEL_REGISTRY[args.model].build_model_from_args(args)


SUPPORTED_MODELS = {
    "complex": "cogdl.models.emb.complex",
    "deepwalk": "cogdl.models.emb.deepwalk",
    "dgk": "cogdl.models.emb.dgk",
    "distmult": "cogdl.models.emb.distmult",
    "dngr": "cogdl.models.emb.dngr",
    "gatne": "cogdl.models.emb.gatne",
    "grarep": "cogdl.models.emb.grarep",
    "graph2vec": "cogdl.models.emb.graph2vec",
    "hin2vec": "cogdl.models.emb.hin2vec",
    "hope": "cogdl.models.emb.hope",
    "line": "cogdl.models.emb.line",
    "metapath2vec": "cogdl.models.emb.metapath2vec",
    "netmf": "cogdl.models.emb.netmf",
    "netsmf": "cogdl.models.emb.netsmf",
    "node2vec": "cogdl.models.emb.node2vec",
    "prone": "cogdl.models.emb.prone",
    "prone++": "cogdl.models.emb.pronepp",
    "pte": "cogdl.models.emb.pte",
    "rotate": "cogdl.models.emb.rotate",
    "sdne": "cogdl.models.emb.sdne",
    "spectral": "cogdl.models.emb.spectral",
    "transe": "cogdl.models.emb.transe",
    "daegc": "cogdl.models.agc.pyg_daegc",
    "agc": "cogdl.models.agc.agc",
    "compgcn": "cogdl.models.nn.compgcn",
    "dgi": "cogdl.models.nn.dgi",
    "disengcn": "cogdl.models.nn.disengcn",
    "gat": "cogdl.models.nn.gat",
    "gcn": "cogdl.models.nn.gcn",
    "gcnii": "cogdl.models.nn.gcnii",
    "gdc_gcn": "cogdl.models.nn.gdc_gcn",
    "grace": "cogdl.models.nn.grace",
    "graphsage": "cogdl.models.nn.graphsage",
    "mvgrl": "cogdl.models.nn.mvgrl",
    "patchy_san": "cogdl.models.nn.patchy_san",
    "ppnp": "cogdl.models.nn.ppnp",
    "rgcn": "cogdl.models.nn.rgcn",
    "sgc": "cogdl.models.nn.sgc",
    "drgat": "cogdl.models.nn.pyg_drgat",
    "gin": "cogdl.models.nn.pyg_gin",
    "infograph": "cogdl.models.nn.pyg_infograph",
    "grand": "cogdl.models.nn.pyg_grand",
    "chebyshev": "cogdl.models.nn.pyg_cheb",
    "unsup_graphsage": "cogdl.models.nn.pyg_unsup_graphsage",
    "hgpsl": "cogdl.models.nn.pyg_hgpsl",
    "gpt_gnn": "cogdl.models.nn.pyg_gpt_gnn",
    "deepergcn": "cogdl.models.nn.pyg_deepergcn",
    "pyg_gat": "cogdl.models.nn.pyg_gat",
    "unet": "cogdl.models.nn.pyg_graph_unet",
    "sortpool": "cogdl.models.nn.pyg_sortpool",
    "pyg_gcn": "cogdl.models.nn.pyg_gcn",
    "mixhop": "cogdl.models.nn.mixhop",
    "pprgo": "cogdl.models.nn.pprgo",
    "dgcnn": "cogdl.models.nn.pyg_dgcnn",
    "infomax": "cogdl.models.nn.pyg_infomax",
    "gtn": "cogdl.models.nn.pyg_gtn",
    "sign": "cogdl.models.nn.pyg_sign",
    "pairnorm": "cogdl.models.nn.pyg_pairnorm",
    "dropedge_gcn": "cogdl.models.nn.dropedge_gcn",
    "fastgcn": "cogdl.models.nn.fastgcn",
    "drgcn": "cogdl.models.nn.pyg_drgcn",
    "gcnmix": "cogdl.models.nn.pyg_gcnmix",
    "mlp": "cogdl.models.nn.mlp",
    "sgcpn": "cogdl.models.nn.sgcpn",
    "stpgnn": "cogdl.models.nn.pyg_stpgnn",
    "pyg_unet": "cogdl.models.nn.pyg_unet",
    "srgcn": "cogdl.models.nn.pyg_srgcn",
    "diffpool": "cogdl.models.nn.pyg_diffpool",
    "asgcn": "cogdl.models.nn.asgcn",
    "han": "cogdl.models.nn.pyg_han",
    "sagpool": "cogdl.models.nn.pyg_sagpool",
}
