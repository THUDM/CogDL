## Usage of GRB part

### 1. Attack

An example of training Graph Convolutional Network ([GCN](https://arxiv.org/abs/1609.02907)) as surrogate model and another GCN as target model on _grb-cora_ dataset and apply FGSM injection attack on surrogate model and target model.

#### 1) Load Dataset

```python
from cogdl.datasets.grb_data import Cora_GRBDataset
dataset = Cora_GRBDataset()
graph = copy.deepcopy(dataset.get(0))
device = "cuda:0"
graph.to(device)
test_mask = graph.test_mask
```

#### 2) Train Surrogate Model

```python
from cogdl.models.nn import GCN
from cogdl.trainer import Trainer
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
import torch
model = GCN(
    in_feats=graph.num_features,
    hidden_size=64,
    out_feats=graph.num_classes,
    num_layers=2,
    dropout=0.5,
    activation=None
)
mw_class = fetch_model_wrapper("node_classification_mw")
dw_class = fetch_data_wrapper("node_classification_dw")
optimizer_cfg = dict(
                    lr=0.01,
                    weight_decay=0
                )
model_wrapper = mw_class(model, optimizer_cfg)
dataset_wrapper = dw_class(dataset)
trainer = Trainer(epochs=2000,
                  early_stopping=True,
                  patience=500,
                  cpu=device=="cpu",
                  device_ids=[0])
trainer.run(model_wrapper, dataset_wrapper)
# load best model
model.load_state_dict(torch.load("./checkpoints/model.pt"), False)
model.to(device)
```

#### 3) Train Target Model

```python
model_target = GCN(
    in_feats=graph.num_features,
    hidden_size=64,
    out_feats=graph.num_classes,
    num_layers=3,
    dropout=0.5,
    activation="relu"
)
mw_class = fetch_model_wrapper("node_classification_mw")
dw_class = fetch_data_wrapper("node_classification_dw")
optimizer_cfg = dict(
                    lr=0.01,
                    weight_decay=0
                )
model_wrapper = mw_class(model_target, optimizer_cfg)
dataset_wrapper = dw_class(dataset)
trainer = Trainer(epochs=2000,
                  early_stopping=True,
                  patience=500,
                  cpu=device=="cpu",
                  device_ids=[0])
trainer.run(model_wrapper, dataset_wrapper)
# load best model
model_target.load_state_dict(torch.load("./checkpoints/model.pt"), False)
model_target.to(device)
```

#### 4) Adversarial attack

```python
# FGSM attack
from attack.injection import FGSM
from cogdl.utils.grb_utils import GCNAdjNorm
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
```

#### 5) Evaluate

```python
from cogdl.utils.grb_utils import evaluate
test_score = evaluate(model,
                      graph,
                      mask=test_mask,
                      device=device)
print("Test score before attack for surrogate model: {:.4f}.".format(test_score))
test_score = evaluate(model, 
                      graph_attack,
                      mask=test_mask,
                      device=device)
print("After attack, test score of surrogate model: {:.4f}".format(test_score))
test_score = evaluate(model_target,
                      graph,
                      mask=test_mask,
                      device=device)
print("Test score before attack for target model: {:.4f}.".format(test_score))
test_score = evaluate(model_target, 
                      graph_attack,
                      mask=test_mask,
                      device=device)
print("After attack, test score of target model: {:.4f}".format(test_score))
```



### 2. Adversarial training

An example of adversarial training for Graph Convolutional Network ([GCN](https://arxiv.org/abs/1609.02907)).

```python
device = "cuda:0"
model = GCN(
    in_feats=graph.num_features,
    hidden_size=64,
    out_feats=graph.num_classes,
    num_layers=3,
    dropout=0.5,
    activation=None,
    norm="layernorm"
)
from attack.injection import FGSM
attack = FGSM(epsilon=0.01,
              n_epoch=10,
              n_inject_max=10,
              n_edge_max=20,
              feat_lim_min=-1,
              feat_lim_max=1,
              device=device,
              verbose=False)
mw_class = fetch_model_wrapper("node_classification_mw")
dw_class = fetch_data_wrapper("node_classification_dw")
optimizer_cfg = dict(
                    lr=0.01,
                    weight_decay=0
                )
model_wrapper = mw_class(model_target, optimizer_cfg)
dataset_wrapper = dw_class(dataset)
# add argument of attack and attack_mode for adversarial training
trainer = Trainer(epochs=200,
                  early_stopping=True,
                  patience=50,
                  cpu=device=="cpu",
                  attack=attack,
                  attack_mode="injection",
                  device_ids=[0])
trainer.run(model_wrapper, dataset_wrapper)
model.load_state_dict(torch.load("./checkpoints/model.pt"), False)
model.to(device)
```



### 3. Defense models

An example of GATGuard (a defense model).

```python
# defnese model: GATGuard
from defense import GATGuard
model = GATGuard(in_feats=graph.num_features,
                        hidden_size=64,
                        out_feats=graph.num_classes,
                        num_layers=3,
                        activation="relu",
                        num_heads=4,
                        drop=True)
print(model)
```



## Todo

- [ ] RobustGCN 存在问题
- [ ] betweenness Flip modification attack 卡住
- [ ] PRBCD modification attack 不支持CUDA，且在cpu上非常慢，输出inf
- [ ] FGSM injection attack 的 GRB 实现似乎采用了迭代梯度下降
- [x] SPEIT injection attack 中 inject_mode = "random-iter" 与 "multi-layer" GRB中似乎没有实现
- [ ] leaderboards