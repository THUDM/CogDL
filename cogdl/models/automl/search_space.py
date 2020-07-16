import torch

class SearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {
                "attention_type": ["node", "edge", "heat", "ppr", "identity",],  # "gaussian"  "layer_sample"
                "activation": ["tanh", "relu", "leaky_relu", "relu6", "linear", "elu"],
                "num_heads": [1, 2, 4, 8, 16],
                "hidden_size": [4, 8, 32, 128],
                "num_hops": [1, 2, 3],
                "normalization": ["row_uniform", "row_softmax", "identity", "symmetry"],
                "adj_normalization": ["identity", "row_uniform", "symmetry"],
                "attention_type_att": ["node", "edge", "heat", "ppr", "identity",], # attention type of the 'softmax' head adjacency matrix 
                # "mlp_layer": [0, 1, 2, 3, 4],
            }
    def get_search_space(self):
        return self.search_space

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    def generate_action_list(self, num_of_layers=1):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list


