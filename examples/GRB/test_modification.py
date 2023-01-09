from cogdl.datasets.grb_data import Cora_GRBDataset
from cogdl.utils import set_random_seed
from cogdl.utils.grb_utils import evaluate, GCNAdjNorm
import copy
import torch
dataset = Cora_GRBDataset()
graph = copy.deepcopy(dataset.get(0))
# device = "cpu"
device = "cuda:0"
device_ids = [0]
graph.to(device)
test_mask = graph.test_mask
set_random_seed(40)


# train surrogate model
from cogdl.models.nn import GCN
from cogdl.trainer import Trainer
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
model = GCN(
    in_feats=graph.num_features,
    hidden_size=64,
    out_feats=graph.num_classes,
    num_layers=2,
    dropout=0.5,
    activation=None
)
print(model)
mw_class = fetch_model_wrapper("node_classification_mw")
dw_class = fetch_data_wrapper("node_classification_dw")
optimizer_cfg = dict(
                    lr=0.01,
                    weight_decay=0
                )
model_wrapper1 = mw_class(model, optimizer_cfg)
dataset_wrapper = dw_class(dataset)
trainer = Trainer(epochs=200,
                  early_stopping=True,
                  patience=50,
                  cpu=device=="cpu",
                  device_ids=[0])
trainer.run(model_wrapper1, dataset_wrapper)
model.load_state_dict(torch.load("./checkpoints/model.pt"), False)
model.to(device)
test_score = evaluate(model,
                      graph,
                      mask=test_mask,
                      device=device)
print("Test score before attack for surrogate model: {:.4f}.".format(test_score))

# train target model

model_target = GCN(
    in_feats=graph.num_features,
    hidden_size=64,
    out_feats=graph.num_classes,
    num_layers=3,
    dropout=0.5,
    activation="relu"
)
print(model_target)
mw_class = fetch_model_wrapper("node_classification_mw")
dw_class = fetch_data_wrapper("node_classification_dw")
optimizer_cfg = dict(
                    lr=0.01,
                    weight_decay=0
                )
model_wrapper = mw_class(model_target, optimizer_cfg)
dataset_wrapper = dw_class(dataset)
trainer = Trainer(epochs=200,
                  early_stopping=True,
                  patience=50,
                  cpu=device=="cpu",
                  device_ids=device_ids)
trainer.run(model_wrapper, dataset_wrapper)
model_target.load_state_dict(torch.load("./checkpoints/model.pt"), False)
model_target.to(device)
test_score = evaluate(model_target,
                      graph,
                      mask=test_mask,
                      device=device)
print("Test score before attack for target model: {:.4f}.".format(test_score))


from attack.modification import PGD
epsilon = 0.1
n_epoch = 500
n_mod_ratio = 0.01
n_node_mod = int(graph.y.shape[0] * n_mod_ratio)
n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
feat_lim_min = 0.0
feat_lim_max = 1.0
early_stop_patience = 200
attack = PGD(epsilon,
             n_epoch,
             n_node_mod,
             n_edge_mod,
             feat_lim_min,
             feat_lim_max,
             early_stop_patience=early_stop_patience,
             device=device)
graph_attack = attack.attack(model, graph)
print(graph_attack)
test_score = evaluate(model, 
                      graph_attack,
                      mask=test_mask,
                      device=device)
print("After attack, test score of surrogate model: {:.4f}".format(test_score))
test_score = evaluate(model_target, 
                      graph_attack,
                      mask=test_mask,
                      device=device)
print("After attack, test score of target model: {:.4f}".format(test_score))