import copy
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.lc_sampler import setup_eval_dataloder, setup_finetune_dataloder, LinearProbingDataLoader
from utils import accuracy, set_random_seed, show_occupied_memory, get_current_lr

import wandb


def linear_probing_minibatch(
    model, graph,
    feats, ego_graph_nodes, labels, 
    lr_f, weight_decay_f, max_epoch_f, 
    device, batch_size=-1, shuffle=True):
    logging.info("-- Linear Probing in downstream tasks ---")
    train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes = ego_graph_nodes
    num_train, num_val = len(train_ego_graph_nodes), len(val_ego_graph_nodes)
    train_lbls, val_lbls, test_lbls = labels
    # if dataset_name in ["ogbn-papers100M", "mag-scholar-f", "mag-scholar-c","ogbn-arxiv","ogbn-products"]:
    # if dataset_name in ["ogbn-papers100M", "mag-scholar-f", "mag-scholar-c", "ogbn-arxiv", "ogbn-products"]:
    eval_loader = setup_eval_dataloder("lc", graph, feats, train_ego_graph_nodes+val_ego_graph_nodes+test_ego_graph_nodes, 512)

    with torch.no_grad():
        model.eval()
        embeddings = []

        for batch in tqdm(eval_loader, desc="Infering..."):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.x
            targets = targets.to(device)
            
            batch_emb = model.embed(batch_g, x)[targets]
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    train_emb, val_emb, test_emb = embeddings[:num_train], embeddings[num_train:num_train+num_val], embeddings[num_train+num_val:]

    batch_size = 5120
    acc = []
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i,_ in enumerate(seeds):
        print(f"####### Run seed {seeds[i]} for LinearProbing...")
        set_random_seed(seeds[i])
        print(f"training sample:{len(train_emb)}")
        test_acc = node_classification_linear_probing(
            (train_emb, val_emb, test_emb), 
            (train_lbls, val_lbls, test_lbls), 
            lr_f, weight_decay_f, max_epoch_f, device, batch_size=batch_size, shuffle=shuffle)
        acc.append(test_acc)

    print(f"# final_acc: {np.mean(acc):.4f}, std: {np.std(acc):.4f}")

    return np.mean(acc)



class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
        

def node_classification_linear_probing(embeddings, labels, lr, weight_decay, max_epoch, device, mute=False, batch_size=-1, shuffle=True):
    criterion = torch.nn.CrossEntropyLoss()

    train_emb, val_emb, test_emb = embeddings
    train_label, val_label, test_label = labels
    train_label = train_label.to(torch.long)
    val_label = val_label.to(torch.long)
    test_label = test_label.to(torch.long)
    
    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    encoder = LogisticRegression(train_emb.shape[1], int(train_label.max().item() + 1))
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    
    if batch_size > 0:
        train_loader = LinearProbingDataLoader(np.arange(len(train_emb)), train_emb, train_label, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=shuffle)
        # train_loader = DataLoader(np.arange(len(train_emb)), batch_size=batch_size, shuffle=False)
        val_loader = LinearProbingDataLoader(np.arange(len(val_emb)), val_emb, val_label, batch_size=batch_size, num_workers=4, persistent_workers=True,shuffle=False)
        test_loader = LinearProbingDataLoader(np.arange(len(test_emb)), test_emb, test_label, batch_size=batch_size, num_workers=4, persistent_workers=True,shuffle=False)
    else:
        train_loader = [np.arange(len(train_emb))]
        val_loader = [np.arange(len(val_emb))]
        test_loader = [np.arange(len(test_emb))]

    def eval_forward(loader, _label):
        pred_all = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            pred = encoder(None, batch_x)
            pred_all.append(pred.cpu())
        pred = torch.cat(pred_all, dim=0)
        acc = accuracy(pred, _label)
        return acc

    for epoch in epoch_iter:
        encoder.train()

        for batch_x, batch_label in train_loader:
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            pred = encoder(None, batch_x)
            loss = criterion(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            encoder.eval()
            val_acc = eval_forward(val_loader, val_label)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(encoder)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}")

    best_model.eval()
    encoder = best_model
    with torch.no_grad():
        test_acc = eval_forward(test_loader, test_label)
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc


def finetune(
    model, 
    graph, 
    feats, 
    ego_graph_nodes, 
    labels, 
    split_idx, 
    lr_f, weight_decay_f, max_epoch_f, 
    use_scheduler, batch_size, 
    device, 
    logger=None, 
    full_graph_forward=False,
):
    logging.info("-- Finetuning in downstream tasks ---")
    train_egs, val_egs, test_egs = ego_graph_nodes
    print(f"num of egos:{len(train_egs)},{len(val_egs)},{len(test_egs)}")

    print(graph.num_nodes())

    train_nid = split_idx["train"].numpy()
    val_nid = split_idx["valid"].numpy()
    test_nid = split_idx["test"].numpy()

    train_lbls, val_lbls, test_lbls = [x.long() for x in labels]
    print(f"num of labels:{len(train_lbls)},{len(val_lbls)},{len(test_lbls)}")

    num_classes = max(max(train_lbls.max().item(), val_lbls.max().item()), test_lbls.max().item()) + 1
    
    model = model.get_encoder()
    model.reset_classifier(int(num_classes))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = setup_finetune_dataloder("lc", graph, feats, train_egs, train_lbls, batch_size=batch_size, shuffle=True)
    val_loader = setup_finetune_dataloder("lc", graph, feats, val_egs, val_lbls, batch_size=batch_size, shuffle=False)
    test_loader = setup_finetune_dataloder("lc", graph, feats, test_egs, test_lbls, batch_size=batch_size, shuffle=False)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr_f, weight_decay=weight_decay_f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_f, weight_decay=weight_decay_f)

    if use_scheduler and max_epoch_f > 0:
        logging.info("Use schedular")
        warmup_epochs = int(max_epoch_f * 0.1)
        # scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch_f) ) * 0.5
        scheduler = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else ( 1 + np.cos((epoch - warmup_epochs) * np.pi / (max_epoch_f - warmup_epochs))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    def eval_with_lc(model, loader):
        pred_counts = []
        model.eval()
        epoch_iter = tqdm(loader)
        with torch.no_grad():
            for batch in epoch_iter:
                batch_g, targets, batch_lbls, node_idx = batch
                batch_g = batch_g.to(device)
                batch_lbls = batch_lbls.to(device)
                x = batch_g.x

                prediction = model(batch_g, x)
                prediction = prediction[targets]
                pred_counts.append((prediction.argmax(1) == batch_lbls))
        pred_counts = torch.cat(pred_counts)
        acc = pred_counts.float().sum() / pred_counts.shape[0]
        return acc
    
    def eval_full_prop(model, g, nfeat, val_nid, test_nid, batch_size, device):
        model.eval()

        with torch.no_grad():
            pred = model.inference(g, nfeat, batch_size, device)
        model.train()

        return accuracy(pred[val_nid], val_lbls.cpu()), accuracy(pred[test_nid], test_lbls.cpu())

    best_val_acc = 0
    best_model = None
    best_epoch = 0
    test_acc = 0
    early_stop_cnt = 0

    for epoch in range(max_epoch_f):
        if epoch == 0:
            scheduler.step()
            continue
        if early_stop_cnt >= 10:
            break
        epoch_iter = tqdm(train_loader)
        losses = []
        model.train()

        for batch_g, targets, batch_lbls, node_idx in epoch_iter:
            batch_g = batch_g.to(device)
            targets = targets.to(device)
            batch_lbls = batch_lbls.to(device)
            x = batch_g.x

            prediction = model(batch_g, x)
            prediction = prediction[targets]
            loss = criterion(prediction, batch_lbls)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            metrics = {"finetune_loss": loss}
            wandb.log(metrics)

            if logger is not None:
                logger.log(metrics)

            epoch_iter.set_description(f"Finetuning | train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if not full_graph_forward:
            if epoch > 0:
                val_acc = eval_with_lc(model, val_loader)
                _test_acc = 0
        else:
            if epoch > 0 and epoch % 1 == 0:
                val_acc, _test_acc = eval_full_prop(model, graph, feats, val_nid, test_nid, 10000, device)
                model = model.to(device)
        
        print('val Acc {:.4f}'.format(val_acc))
        if val_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = val_acc
            test_acc = _test_acc
            best_epoch = epoch
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if not full_graph_forward:
            print("val Acc {:.4f}, Best Val Acc {:.4f}".format(val_acc, best_val_acc))
        else:
            print("Val Acc {:.4f}, Best Val Acc {:.4f} Test Acc {:.4f}".format(val_acc, best_val_acc, test_acc))

        metrics = {"epoch_val_acc": val_acc,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "lr_f": get_current_lr(optimizer)}

        wandb.log(metrics)
        if logger is not None:
            logger.log(metrics)
        print(f"# Finetuning - Epoch {epoch} | train_loss: {np.mean(losses):.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, Memory: {show_occupied_memory():.2f} MB")

    model = best_model
    if not full_graph_forward:
        test_acc = eval_with_lc(test_loader)

    print(f"Finetune | TestAcc: {test_acc:.4f} from Epoch {best_epoch}")
    return test_acc


def linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    with torch.no_grad():
        x = model.embed(graph.to(device), x.to(device))
        in_feat = x.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = torch.optim.Adam(encoder.parameters(), lr=lr_f, weight_decay=weight_decay_f)
    final_acc, estp_acc = _linear_probing_full_batch(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc


def _linear_probing_full_batch(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    labels = graph.y

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc, estp_test_acc
