from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper


def set_default_wrapper_config():
    node_classification_models = [
        "gcn",
        "deepergcn",
        "drgcn",
        "drgat",
        "gcnii",
        "gcnmix",
        "grand",
        "grace",
        "mvgrl",
        "graphsage",
        "sage",
        "gdc_gcn",
        "mixhop",
        "mlp",
        "moe_gcn",
        "ppnp",
        "appnp",
        "pprgo",
        "chebyshev",
        "unet",
        "srgcn",
        "revgcn",
        "revgat",
        "revgen",
        "sagn",
        "sign",
        "sgc",
        "unsup_graphsage",
        "dgi",
        "dropedge_gcn",
        "gat",
        "graphsaint",
        "m3s",
        "correct_smooth_mlp",
    ]

    graph_classification_models = ["gin", "patchy_san", "diffpool", "infograph", "dgcnn", "sortpool"]

    network_embedding_models = [
        "deepwalk",
        "line",
        "node2vec",
        "prone",
        "netmf",
        "netsmf",
        "sdne",
        "spectral",
        "dngr",
        "grarep",
        "hope",
    ]

    graph_embedding_models = [
        "dgk",
        "graph2vec",
    ]

    graph_clustering_models = [
        "agc",
        "daegc",
        "gae",
        "vgae",
    ]

    graph_kg_link_prediction = ["rgcn", "compgcn"]

    heterogeneous_gnn_models = [
        "gtn",
        "han",
    ]

    heterogeneous_emb_models = [
        "metapath2vec",
        "pte",
        "hin2vec",
    ]
    triple_link_prediction_models=["transe","distmult", "rotate", "complex"]
    triple_link_prediction_wrappers=dict()
    for item in triple_link_prediction_models:
        triple_link_prediction_wrappers[item] = {"mw": "triple_link_prediction_mw", "dw": "triple_link_prediction_dw"}


    node_classification_wrappers = dict()
    for item in node_classification_models:
        node_classification_wrappers[item] = {"mw": "node_classification_mw", "dw": "node_classification_dw"}

    node_classification_wrappers["dgi"]["mw"] = "dgi_mw"
    node_classification_wrappers["m3s"]["mw"] = "m3s_mw"
    node_classification_wrappers["graphsage"]["mw"] = "graphsage_mw"
    node_classification_wrappers["unsup_graphsage"]["mw"] = "unsup_graphsage_mw"
    node_classification_wrappers["mvgrl"]["mw"] = "mvgrl_mw"
    node_classification_wrappers["sagn"]["mw"] = "sagn_mw"
    node_classification_wrappers["grand"]["mw"] = "grand_mw"
    node_classification_wrappers["gcnmix"]["mw"] = "gcnmix_mw"
    node_classification_wrappers["grace"]["mw"] = "grace_mw"
    node_classification_wrappers["pprgo"]["mw"] = "pprgo_mw"

    node_classification_wrappers["m3s"]["dw"] = "m3s_dw"
    node_classification_wrappers["graphsage"]["dw"] = "graphsage_dw"
    node_classification_wrappers["unsup_graphsage"]["dw"] = "unsup_graphsage_dw"
    node_classification_wrappers["pprgo"]["dw"] = "pprgo_dw"
    node_classification_wrappers["sagn"]["dw"] = "sagn_dw"

    graph_classification_wrappers = dict()
    for item in graph_classification_models:
        graph_classification_wrappers[item] = {"mw": "graph_classification_mw", "dw": "graph_classification_dw"}

    graph_classification_wrappers["infograph"] = {"mw": "infograph_mw", "dw": "infograph_dw"}

    network_embedding_wrappers = dict()
    for item in network_embedding_models:
        network_embedding_wrappers[item] = {"mw": "network_embedding_mw", "dw": "network_embedding_dw"}

    graph_embedding_wrappers = dict()
    for item in graph_embedding_models:
        graph_embedding_wrappers[item] = {"mw": "graph_embedding_mw", "dw": "graph_embedding_dw"}

    graph_clustering_wrappers = dict()
    for item in graph_clustering_models:
        graph_clustering_wrappers[item] = {"dw": "node_classification_dw"}
    graph_clustering_wrappers["gae"]["mw"] = "gae_mw"
    graph_clustering_wrappers["vgae"]["mw"] = "gae_mw"
    graph_clustering_wrappers["agc"]["mw"] = "agc_mw"
    graph_clustering_wrappers["daegc"]["mw"] = "daegc_mw"

    graph_kg_link_prediction_wrappers = dict()
    for item in graph_kg_link_prediction:
        graph_kg_link_prediction_wrappers[item] = {"dw": "gnn_kg_link_prediction_dw", "mw": "gnn_kg_link_prediction_mw"}

    heterogeneous_gnn_wrappers = dict()
    for item in heterogeneous_gnn_models:
        heterogeneous_gnn_wrappers[item] = {"dw": "heterogeneous_gnn_dw", "mw": "heterogeneous_gnn_mw"}

    heterogeneous_emb_wrappers = dict()
    for item in heterogeneous_emb_models:
        heterogeneous_emb_wrappers[item] = {"dw": "heterogeneous_embedding_dw", "mw": "heterogeneous_embedding_mw"}

    traffic_prediction_models = ["stgcn", "stgat"]
    traffic_prediction_wrappers = dict()
    for item in traffic_prediction_models:
        traffic_prediction_wrappers[item] = {"dw": "{}_dw".format(item), "mw": "{}_mw".format(item)}

    other_wrappers = dict()
    other_wrappers["gatne"] = {"mw": "multiplex_embedding_mw", "dw": "multiplex_embedding_dw"}
    other_wrappers["gcc"] = {"mw": "gcc_mw", "dw": "gcc_dw"}

    merged = dict()
    merged.update(node_classification_wrappers)
    merged.update(graph_embedding_wrappers)
    merged.update(graph_classification_wrappers)
    merged.update(network_embedding_wrappers)
    merged.update(graph_clustering_wrappers)
    merged.update(graph_kg_link_prediction_wrappers)
    merged.update(heterogeneous_gnn_wrappers)
    merged.update(heterogeneous_emb_wrappers)
    merged.update(other_wrappers)
    merged.update(triple_link_prediction_wrappers)
    merged.update(traffic_prediction_wrappers)
    return merged


default_wrapper_config = set_default_wrapper_config()


def get_wrappers(model_name):
    if model_name in default_wrapper_config:
        dw = default_wrapper_config[model_name]["dw"]
        mw = default_wrapper_config[model_name]["mw"]
        return fetch_model_wrapper(mw), fetch_data_wrapper(dw)


def get_wrappers_name(model_name):
    if model_name in default_wrapper_config:
        dw = default_wrapper_config[model_name]["dw"]
        mw = default_wrapper_config[model_name]["mw"]
        return mw, dw
