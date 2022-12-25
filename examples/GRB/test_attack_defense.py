from cogdl.datasets.grb_data import Cora_GRBDataset
from attack.modification import *
from attack.injection import *
from attack.modification import RAND as RAND_Modify
from attack.injection import RAND as RAND_Inject
from attack.modification import PGD as PGD_Modify
from attack.injection import PGD as PGD_Inject
from defense import *
from cogdl.models.nn import GCN
from cogdl.utils import set_random_seed
from cogdl.utils.grb_utils import evaluate, GCNAdjNorm
from cogdl.trainer import Trainer
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
import copy
import torch


def train_model(model, graph, dataset, test_mask, device, device_ids):
    print(model)
    print("training...")
    mw_class = fetch_model_wrapper("node_classification_mw")
    dw_class = fetch_data_wrapper("node_classification_dw")
    optimizer_cfg = dict(
                        lr=0.01,
                        weight_decay=0
                    )
    model_wrapper1 = mw_class(model, optimizer_cfg)
    dataset_wrapper = dw_class(dataset)
    trainer = Trainer(epochs=3,
                    early_stopping=True,
                    patience=1,
                    cpu=device=="cpu",
                    device_ids=device_ids)
    trainer.run(model_wrapper1, dataset_wrapper)
    model.load_state_dict(torch.load("./checkpoints/model.pt"), False)
    model.to(device)
    test_score = evaluate(model,
                        graph,
                        mask=test_mask,
                        device=device)
    return test_score


def init_dataset():
    dataset = Cora_GRBDataset()
    graph = copy.deepcopy(dataset.get(0))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_ids = [0]
    graph.to(device)
    test_mask = graph.test_mask
    set_random_seed(40)
    return graph, dataset, test_mask, device, device_ids


def init_surrogate_model(graph, dataset, test_mask, device, device_ids):
    model_sur = GCN(
        in_feats=graph.num_features,
        hidden_size=16,
        out_feats=graph.num_classes,
        num_layers=2,
        dropout=0.5,
        activation=None
    )
    score_sur = train_model(model_sur, graph, dataset, test_mask, device, device_ids)
    print("Test score before attack for surrogate model: {:.4f}.".format(score_sur))
    return model_sur


def init_target_model(graph, dataset, test_mask, device, device_ids):
    model_target = GCN(
        in_feats=graph.num_features,
        hidden_size=16,
        out_feats=graph.num_classes,
        num_layers=3,
        dropout=0.5,
        activation="relu"
    )
    score_target = train_model(model_target, graph, dataset, test_mask, device, device_ids)
    print("Test score before attack for target model: {:.4f}.".format(score_target))
    return model_target


def test_dice_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("DICE modification attack...")
    n_mod_ratio = 0.01
    ratio_delete = 0.02
    attack = DICE(int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio),
                ratio_delete,
                device=device)
    graph_attack = attack.attack(graph)
    test_score = evaluate(model_sur, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of surrogate model: {:.4f}".format(test_score))
    test_score = evaluate(model_target, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of target model: {:.4f}".format(test_score))


def test_fga_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("FGA modification attack...")
    n_mod_ratio = 0.01
    attacks = [
        FGA(int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio), allow_isolate=False, device=device),
        FGA(int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio), allow_isolate=True, device=device)
    ]
    for attack in attacks:
        graph_attack = attack.attack(model_sur, graph)
        test_score = evaluate(model_sur, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After attack, test score of surrogate model: {:.4f}".format(test_score))
        test_score = evaluate(model_target, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After attack, test score of surrogate model: {:.4f}".format(test_score))


def test_flip_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("FLIP modification attack...")
    n_mod_ratio = 0.01
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    attacks = {
        "degree_flip": FLIP(n_edge_mod, flip_type="deg", mode="descend", device=device),    # degree flipping
        "eigen_flip": FLIP(n_edge_mod, flip_type="eigen", mode="descend", device=device),   # eigen flipping
        # "betweenness_flip": FLIP(n_edge_mod, flip_type="bet", mode="ascend", device=device) # betweenness flipping
    }
    for key, attack in attacks.items():
        graph_attack = attack.attack(graph)
        print(graph_attack)
        test_score = evaluate(model_sur, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After {} attack, test score of surrogate model: {:.4f}".format(key, test_score))
        test_score = evaluate(model_target, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After {} attack, test score of target model: {:.4f}".format(key, test_score))


def test_rand_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("RAND modification attack...")
    n_mod_ratio = 0.01
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    attacks = [
        RAND_Modify(n_edge_mod, allow_isolate=False, device=device),
        RAND_Modify(n_edge_mod, allow_isolate=True, device=device)
    ]
    for attack in attacks:
        graph_attack = attack.attack(graph)
        print(graph_attack)
        test_score = evaluate(model_sur, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After attack, test score of surrogate model: {:.4f}".format(test_score))
        test_score = evaluate(model_target, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After attack, test score of target model: {:.4f}".format(test_score))


def test_nea_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("NEA modification attack...")
    n_mod_ratio = 0.01
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    attack = NEA(n_edge_mod, device=device)
    graph_attack = attack.attack(graph)
    print(graph_attack)
    test_score = evaluate(model_sur, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of surrogate model: {:.4f}".format(test_score))
    test_score = evaluate(model_target, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of target model: {:.4f}".format(test_score))


def test_stack_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("STACK modification attack...")
    n_mod_ratio = 0.01
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    attacks = [
        STACK(n_edge_mod, allow_isolate=False, device=device),
        STACK(n_edge_mod, allow_isolate=True, device=device)
    ]
    for attack in attacks:
        graph_attack = attack.attack(graph)
        print(graph_attack)
        test_score = evaluate(model_sur, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After attack, test score of surrogate model: {:.4f}".format(test_score))
        test_score = evaluate(model_target, 
                            graph_attack,
                            mask=test_mask,
                            device=device)
        print("After attack, test score of target model: {:.4f}".format(test_score))


def test_pgd_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("PGD modification attack...")
    epsilon = 0.1
    n_epoch = 5
    n_mod_ratio = 0.01
    n_node_mod = int(graph.y.shape[0] * n_mod_ratio)
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    feat_lim_min = 0.0
    feat_lim_max = 1.0
    early_stop_patience = 2
    attack = PGD_Modify(epsilon,
                n_epoch,
                n_node_mod,
                n_edge_mod,
                feat_lim_min,
                feat_lim_max,
                early_stop=True,
                early_stop_patience=early_stop_patience,
                early_stop_epsilon=1e-3,
                device=device)
    graph_attack = attack.attack(model_sur, graph)
    print(graph_attack)
    test_score = evaluate(model_sur, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of surrogate model: {:.4f}".format(test_score))
    test_score = evaluate(model_target, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of target model: {:.4f}".format(test_score))


def test_prbcd_modification_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("PRBCD modification attack...")
    device = "cpu"
    epsilon = 0.3
    n_epoch = 5
    n_mod_ratio = 0.01
    n_node_mod = int(graph.y.shape[0] * n_mod_ratio)
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    feat_lim_min = 0.0
    feat_lim_max = 1.0
    early_stop_patience = 2
    early_stop_epsilon = 1e-3
    attack = PRBCD(epsilon,
                n_epoch,
                n_node_mod,
                n_edge_mod,
                feat_lim_min,
                feat_lim_max,
                early_stop=True,
                early_stop_patience=early_stop_patience,
                early_stop_epsilon=early_stop_epsilon,
                device=device)
    graph_attack = attack.attack(model_sur, graph)
    print(graph_attack)
    test_score = evaluate(model_sur, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of surrogate model: {:.4f}".format(test_score))
    test_score = evaluate(model_target, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of target model: {:.4f}".format(test_score))


def test_fgsm_injection_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("FGSM injection attack...")
    attack = FGSM(epsilon=0.01,
                n_epoch=5,
                n_inject_max=10,
                n_edge_max=20,
                feat_lim_min=-1,
                feat_lim_max=1,
                early_stop=True,
                early_stop_patience=2,
                early_stop_epsilon=1e-4,
                device=device)
    graph_attack = attack.attack(model=model_sur,
                                graph=graph,
                                adj_norm_func=GCNAdjNorm)
    print(graph_attack)
    test_score_sur = evaluate(model_sur,
                            graph_attack,
                            mask=test_mask,
                            device=device)
    print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))
    test_score_target_attack = evaluate(model_target,
                                        graph_attack,
                                        mask=test_mask,
                                        device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))


def test_pgd_injection_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("PGD injection attack...")
    attack = PGD_Inject(epsilon=0.01,
                n_epoch=5,
                n_inject_max=10,
                n_edge_max=20,
                feat_lim_min=-1,
                feat_lim_max=1,
                early_stop=True,
                early_stop_patience=2,
                early_stop_epsilon=1e-4,
                device=device)
    graph_attack = attack.attack(model=model_sur,
                                graph=graph,
                                adj_norm_func=GCNAdjNorm)
    print(graph_attack)
    test_score_sur = evaluate(model_sur,
                            graph_attack,
                            mask=test_mask,
                            device=device)
    print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))
    test_score_target_attack = evaluate(model_target,
                                        graph_attack,
                                        mask=test_mask,
                                        device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))


def test_rand_injection_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("RAND injection attack...")
    attack = RAND_Inject(n_inject_max=10,
                n_edge_max=20,
                feat_lim_min=-1,
                feat_lim_max=1,
                device=device)
    graph_attack = attack.attack(model=model_sur,
                                graph=graph,
                                adj_norm_func=GCNAdjNorm)
    print(graph_attack)
    test_score_sur = evaluate(model_sur,
                            graph_attack,
                            mask=test_mask,
                            device=device)
    print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))
    test_score_target_attack = evaluate(model_target,
                                        graph_attack,
                                        mask=test_mask,
                                        device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))


def test_speit_injection_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("SPEIT injection attack...")
    inject_modes = ["random", "random-iter", "multi-layer"]
    attacks = [SPEIT(lr=0.02,
                n_epoch=5,
                n_inject_max=10,
                n_edge_max=20,
                feat_lim_min=-1,
                feat_lim_max=1,
                inject_mode=inject_mode,
                early_stop=True,
                early_stop_patience=2,
                early_stop_epsilon=1e-4,
                device=device)
                for inject_mode in inject_modes]
    for attack in attacks:
        graph_attack = attack.attack(model=model_sur,
                                    graph=graph,
                                    adj_norm_func=GCNAdjNorm)
        print(graph_attack)
        test_score_sur = evaluate(model_sur,
                                graph_attack,
                                mask=test_mask,
                                device=device)
        print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))
        test_score_target_attack = evaluate(model_target,
                                            graph_attack,
                                            mask=test_mask,
                                            device=device)
        print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))


def test_tdgia_injection_attack():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = init_target_model(graph, dataset, test_mask, device, device_ids)
    print("TDGIA injection attack...")
    inject_modes = ["tdgia", "random", "uniform"]
    attacks = []
    for inject_mode in inject_modes:
        attacks.append(TDGIA(lr=0.01,
            n_epoch=5,
            n_inject_max=10,
            n_edge_max=20,
            feat_lim_min=-1,
            feat_lim_max=1,
            inject_mode=inject_mode,
            early_stop=True,
            early_stop_patience=2,
            early_stop_epsilon=1e-4,
            device=device)
        )
    for attack in attacks:
        graph_attack = attack.attack(model=model_sur,
                                    graph=graph,
                                    adj_norm_func=GCNAdjNorm)
        print(graph_attack)
        test_score_sur = evaluate(model_sur,
                                graph_attack,
                                mask=test_mask,
                                device=device)
        print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))
        test_score_target_attack = evaluate(model_target,
                                            graph_attack,
                                            mask=test_mask,
                                            device=device)
        print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))


def apply_pgd_modification_attack(model_sur, model_target, graph, test_mask, device):
    print("PGD modification attack...")
    epsilon = 0.1
    n_epoch = 5
    n_mod_ratio = 0.01
    n_node_mod = int(graph.y.shape[0] * n_mod_ratio)
    n_edge_mod = int(graph.to_scipy_csr()[test_mask.cpu()].getnnz() * n_mod_ratio)
    feat_lim_min = 0.0
    feat_lim_max = 1.0
    early_stop_patience = 2
    attack = PGD_Modify(epsilon,
                n_epoch,
                n_node_mod,
                n_edge_mod,
                feat_lim_min,
                feat_lim_max,
                early_stop_patience=early_stop_patience,
                device=device)
    graph_attack = attack.attack(model_sur, graph)
    print(graph_attack)
    test_score = evaluate(model_sur, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of surrogate model: {:.4f}".format(test_score))
    test_score = evaluate(model_target, 
                        graph_attack,
                        mask=test_mask,
                        device=device)
    print("After attack, test score of target model: {:.4f}".format(test_score))


def test_gcnsvd_defense():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    print("GCNSVD defense model...")
    model_target = GCNSVD(
        in_feats=graph.num_features,
        hidden_size=16,
        out_feats=graph.num_classes,
        num_layers=2,
        dropout=0.5,
        activation="relu",
        adj_norm_func=GCNAdjNorm,
        norm = "layernorm"
    )
    test_score = train_model(model_target, graph, dataset, test_mask, device, device_ids)
    print("Test score before attack for target model: {:.4f}.".format(test_score))
    apply_pgd_modification_attack(model_sur, model_target, graph, test_mask, device)


def test_gatguard_defense():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    print("GATGuard defense model...")
    model_target = GATGuard(in_feats=graph.num_features,
        hidden_size=16,
        out_feats=graph.num_classes,
        num_layers=2,
        activation="relu",
        num_heads=4,
        drop=True
    )
    test_score = train_model(model_target, graph, dataset, test_mask, device, device_ids)
    print("Test score before attack for target model: {:.4f}.".format(test_score))
    apply_pgd_modification_attack(model_sur, model_target, graph, test_mask, device)


def test_gcnguard_defense():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    print("GATGuard defense model...")
    model_target = GCNGuard(in_feats=graph.num_features,
        hidden_size=16,
        out_feats=graph.num_classes,
        num_layers=2,
        activation="relu",
        norm="layernorm",
        drop=True
    )
    test_score = train_model(model_target, graph, dataset, test_mask, device, device_ids)
    print("Test score before attack for target model: {:.4f}.".format(test_score))
    apply_pgd_modification_attack(model_sur, model_target, graph, test_mask, device)


def test_robustgcn_defense():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    print("RobustGCN defense model...")
    model_target = RobustGCN(in_feats=graph.num_features,
        hidden_size=16,
        dropout=0.5,
        out_feats=graph.num_classes,
        num_layers=2
    )
    test_score = train_model(model_target, graph, dataset, test_mask, device, device_ids)
    print("Test score before attack for target model: {:.4f}.".format(test_score))
    apply_pgd_modification_attack(model_sur, model_target, graph, test_mask, device)


def test_adversarial_train():
    graph, dataset, test_mask, device, device_ids = init_dataset()
    model_sur = init_surrogate_model(graph, dataset, test_mask, device, device_ids)
    model_target = GCN(
        in_feats=graph.num_features,
        hidden_size=16,
        out_feats=graph.num_classes,
        num_layers=3,
        dropout=0.5,
        activation=None,
        norm="layernorm"
    )
    attack = FGSM(epsilon=0.01,
        n_epoch=5,
        n_inject_max=10,
        n_edge_max=20,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device,
        verbose=False
    )
    mw_class = fetch_model_wrapper("node_classification_mw")
    dw_class = fetch_data_wrapper("node_classification_dw")
    optimizer_cfg = dict(
                        lr=0.01,
                        weight_decay=0
                    )
    model_wrapper = mw_class(model_target, optimizer_cfg)
    dataset_wrapper = dw_class(dataset)
    trainer = Trainer(epochs=3,
                    early_stopping=True,
                    patience=1,
                    cpu=device=="cpu",
                    attack=attack,
                    attack_mode="injection",
                    device_ids=device_ids)
    trainer.run(model_wrapper, dataset_wrapper)
    model_target.load_state_dict(torch.load("./checkpoints/model.pt"), False)
    model_target.to(device)
    test_score = evaluate(model_target,
                        graph,
                        mask=test_mask,
                        device=device)
    print("Test score before attack for target model: {:.4f}.".format(test_score))
    graph_attack = attack.attack(model=model_sur,
                                graph=graph,
                                adj_norm_func=GCNAdjNorm)
    print(graph_attack)
    test_score_sur = evaluate(model_sur,
                            graph_attack,
                            mask=test_mask,
                            device=device)
    print("Test score after attack for surrogate model: {:.4f}.".format(test_score_sur))
    test_score_target_attack = evaluate(model_target,
                                        graph_attack,
                                        mask=test_mask,
                                        device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score_target_attack))


if __name__ == "__main__":
    test_dice_modification_attack()
    test_fga_modification_attack()
    test_flip_modification_attack()
    test_rand_modification_attack()
    test_nea_modification_attack()
    test_stack_modification_attack()
    test_pgd_modification_attack()
    test_prbcd_modification_attack()
    test_fgsm_injection_attack()
    test_pgd_injection_attack()
    test_rand_injection_attack()
    test_speit_injection_attack()
    test_tdgia_injection_attack()
    test_gcnsvd_defense()
    test_gatguard_defense()
    test_gcnguard_defense()
    test_robustgcn_defense()
    test_adversarial_train()
