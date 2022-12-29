from .base_data_wrapper import DataWrapper
import importlib


def register_data_wrapper(name):
    """
    New data wrapper types can be added to cogdl with the :func:`register_data_wrapper`
    function decorator.

    Args:
        name (str): the name of the data_wrapper
    """

    def register_data_wrapper_cls(cls):
        print("The `register_data_wrapper` API is deprecated!")
        return cls

    return register_data_wrapper_cls


def fetch_data_wrapper(name):
    if isinstance(name, type):
        return name
    if name in SUPPORTED_DW:
        path = ".".join(SUPPORTED_DW[name].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        raise NotImplementedError(f"Failed to import {name} DataWrapper.")
    class_name = SUPPORTED_DW[name].split(".")[-1]
    return getattr(module, class_name)


SUPPORTED_DW = {
    "triple_link_prediction_dw": "cogdl.wrappers.data_wrapper.link_prediction.TripleDataWrapper",
    "cluster_dw": "cogdl.wrappers.data_wrapper.node_classification.ClusterWrapper",
    "graphsage_dw": "cogdl.wrappers.data_wrapper.node_classification.GraphSAGEDataWrapper",
    "unsup_graphsage_dw": "cogdl.wrappers.data_wrapper.node_classification.UnsupGraphSAGEDataWrapper",
    "m3s_dw": "cogdl.wrappers.data_wrapper.node_classification.M3SDataWrapper",
    "network_embedding_dw": "cogdl.wrappers.data_wrapper.node_classification.NetworkEmbeddingDataWrapper",
    "node_classification_dw": "cogdl.wrappers.data_wrapper.node_classification.FullBatchNodeClfDataWrapper",
    "pprgo_dw": "cogdl.wrappers.data_wrapper.node_classification.PPRGoDataWrapper",
    "sagn_dw": "cogdl.wrappers.data_wrapper.node_classification.SAGNDataWrapper",
    "gcc_dw": "cogdl.wrappers.data_wrapper.pretraining.GCCDataWrapper",
    "embedding_link_prediction_dw": "cogdl.wrappers.data_wrapper.link_prediction.EmbeddingLinkPredictionDataWrapper",
    "gnn_kg_link_prediction_dw": "cogdl.wrappers.data_wrapper.link_prediction.GNNKGLinkPredictionDataWrapper",
    "gnn_link_prediction_dw": "cogdl.wrappers.data_wrapper.link_prediction.GNNLinkPredictionDataWrapper",
    "heterogeneous_embedding_dw": "cogdl.wrappers.data_wrapper.heterogeneous.HeterogeneousEmbeddingDataWrapper",
    "heterogeneous_gnn_dw": "cogdl.wrappers.data_wrapper.heterogeneous.HeterogeneousGNNDataWrapper",
    "multiplex_embedding_dw": "cogdl.wrappers.data_wrapper.heterogeneous.MultiplexEmbeddingDataWrapper",
    "graph_classification_dw": "cogdl.wrappers.data_wrapper.graph_classification.GraphClassificationDataWrapper",
    "graph_embedding_dw": "cogdl.wrappers.data_wrapper.graph_classification.GraphEmbeddingDataWrapper",
    "infograph_dw": "cogdl.wrappers.data_wrapper.graph_classification.InfoGraphDataWrapper",
    "patchy_san_dw": "cogdl.wrappers.data_wrapper.graph_classification.PATCHY_SAN_DataWrapper",
    "stgcn_dw": "cogdl.wrappers.data_wrapper.traffic_prediction.STGCNDataWrapper",
    "stgat_dw": "cogdl.wrappers.data_wrapper.traffic_prediction.STGATDataWrapper",
}
