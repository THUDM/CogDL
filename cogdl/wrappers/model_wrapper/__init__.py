import importlib

from .base_model_wrapper import ModelWrapper, EmbeddingModelWrapper, UnsupervisedModelWrapper


def register_model_wrapper(name):
    """
    New data wrapper types can be added to cogdl with the :func:`register_model_wrapper`
    function decorator.

    Args:
        name (str): the name of the model_wrapper
    """

    def register_model_wrapper_cls(cls):
        print("The `register_model_wrapper` API is deprecated!")
        return cls

    return register_model_wrapper_cls


def fetch_model_wrapper(name):
    if isinstance(name, type):
        return name
    if name in SUPPORTED_MW:
        path = ".".join(SUPPORTED_MW[name].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        raise NotImplementedError(f"Failed to import {name} ModelWrapper.")
    class_name = SUPPORTED_MW[name].split(".")[-1]
    return getattr(module, class_name)


SUPPORTED_MW = {
    "triple_link_prediction_mw": "cogdl.wrappers.model_wrapper.link_prediction.TripleModelWrapper",
    "dgi_mw": "cogdl.wrappers.model_wrapper.node_classification.DGIModelWrapper",
    "gcnmix_mw": "cogdl.wrappers.model_wrapper.node_classification.GCNMixModelWrapper",
    "grace_mw": "cogdl.wrappers.model_wrapper.node_classification.GRACEModelWrapper",
    "grand_mw": "cogdl.wrappers.model_wrapper.node_classification.GrandModelWrapper",
    "mvgrl_mw": "cogdl.wrappers.model_wrapper.node_classification.MVGRLModelWrapper",
    "self_auxiliary_mw": "cogdl.wrappers.model_wrapper.node_classification.SelfAuxiliaryModelWrapper",
    "graphsage_mw": "cogdl.wrappers.model_wrapper.node_classification.GraphSAGEModelWrapper",
    "unsup_graphsage_mw": "cogdl.wrappers.model_wrapper.node_classification.UnsupGraphSAGEModelWrapper",
    "m3s_mw": "cogdl.wrappers.model_wrapper.node_classification.M3SModelWrapper",
    "network_embedding_mw": "cogdl.wrappers.model_wrapper.node_classification.NetworkEmbeddingModelWrapper",
    "node_classification_mw": "cogdl.wrappers.model_wrapper.node_classification.NodeClfModelWrapper",
    "correct_smooth_mw": "cogdl.wrappers.model_wrapper.node_classification.CorrectSmoothModelWrapper",
    "pprgo_mw": "cogdl.wrappers.model_wrapper.node_classification.PPRGoModelWrapper",
    "sagn_mw": "cogdl.wrappers.model_wrapper.node_classification.SAGNModelWrapper",
    "gcc_mw": "cogdl.wrappers.model_wrapper.pretraining.GCCModelWrapper",
    "embedding_link_prediction_mw": "cogdl.wrappers.model_wrapper.link_prediction.EmbeddingLinkPredictionModelWrapper",
    "gnn_kg_link_prediction_mw": "cogdl.wrappers.model_wrapper.link_prediction.GNNKGLinkPredictionModelWrapper",
    "gnn_link_prediction_mw": "cogdl.wrappers.model_wrapper.link_prediction.GNNLinkPredictionModelWrapper",
    "heterogeneous_embedding_mw": "cogdl.wrappers.model_wrapper.heterogeneous.HeterogeneousEmbeddingModelWrapper",
    "heterogeneous_gnn_mw": "cogdl.wrappers.model_wrapper.heterogeneous.HeterogeneousGNNModelWrapper",
    "multiplex_embedding_mw": "cogdl.wrappers.model_wrapper.heterogeneous.MultiplexEmbeddingModelWrapper",
    "graph_classification_mw": "cogdl.wrappers.model_wrapper.graph_classification.GraphClassificationModelWrapper",
    "graph_embedding_mw": "cogdl.wrappers.model_wrapper.graph_classification.GraphEmbeddingModelWrapper",
    "infograph_mw": "cogdl.wrappers.model_wrapper.graph_classification.InfoGraphModelWrapper",
    "agc_mw": "cogdl.wrappers.model_wrapper.clustering.AGCModelWrapper",
    "daegc_mw": "cogdl.wrappers.model_wrapper.clustering.DAEGCModelWrapper",
    "gae_mw": "cogdl.wrappers.model_wrapper.clustering.GAEModelWrapper",
    "stgcn_mw": "cogdl.wrappers.model_wrapper.traffic_prediction.STGCNModelWrapper",
    "stgat_mw": "cogdl.wrappers.model_wrapper.traffic_prediction.STGATModelWrapper",
}
