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
model_sur = GCN(
    in_feats=graph.num_features,
    hidden_size=64,
    out_feats=graph.num_classes,
    num_layers=2,
    dropout=0.5,
    activation="relu"
)
print(model_sur)
mw_class = fetch_model_wrapper("node_classification_mw")
dw_class = fetch_data_wrapper("node_classification_dw")
optimizer_cfg = dict(
                    lr=0.01,
                    weight_decay=0
                )
model_wrapper = mw_class(model_sur, optimizer_cfg)
dataset_wrapper = dw_class(dataset)
trainer = Trainer(epochs=200,
                  early_stopping=True,
                  patience=50,
                  cpu=device=="cpu",
                  device_ids=device_ids)
trainer.run(model_wrapper, dataset_wrapper)
model_sur.load_state_dict(torch.load("./checkpoints/model.pt"), False)
model_sur.to(device)
test_score = evaluate(model_sur,
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
    activation=None,
    norm="layernorm"
)
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


# FGSM attack
from attack.injection import FGSM
attack = FGSM(epsilon=0.01,
              n_epoch=1000,
              n_inject_max=100,
              n_edge_max=200,
              feat_lim_min=-1,
              feat_lim_max=1,
              device=device)
graph_attack = attack.attack(model=model_sur,
                             graph=graph,
                             adj_norm_func=GCNAdjNorm)

# apply injection attack
test_score_sur = evaluate(model_sur,
                          graph_attack,
                          mask=test_mask,
                          device=device)
print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))

# transfer to target model
test_score_target_attack = evaluate(model_target,
                                    graph_attack,
                                    mask=test_mask,
                                    device=device)
print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))