
import numpy as np
import torch
from sklearn.metrics import f1_score

import logging
import yaml
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
)
from graphmae.datasets.data_util import load_inductive_dataset
from graphmae.models import build_model
from graphmae.evaluation import linear_probing_for_inductive_node_classiifcation, LogisticRegression


def evaluete(model, loaders, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        if len(loaders[0][0].y.shape) > 1: # 原方法用对图和单图区别ppi数据集，这里不适用，换成标签的维度
            x_all = {"train": [], "val": [], "test": []}
            y_all = {"train": [], "val": [], "test": []}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.x
                        x = model.embed(subgraph, feat)
                        x_all[key].append(x)
                        y_all[key].append(subgraph.y)  
            in_dim = x_all["train"][0].shape[1]
            encoder = LogisticRegression(in_dim, num_classes)
            num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
            if not mute:
                print(f"num parameters for finetuning: {sum(num_finetune_params)}")
                # torch.save(x.cpu(), "feat.pt")
            
            encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
            final_acc, estp_acc = mutli_graph_linear_evaluation(encoder, x_all, y_all, optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc
        else:
            x_all = {"train": None, "val": None, "test": None}
            y_all = {"train": None, "val": None, "test": None}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.x
                        x = model.embed(subgraph, feat)
                        mask = getattr(subgraph, f"{key}_mask")
                        x_all[key] = x[mask]
                        y_all[key] = subgraph.y[mask]  
            in_dim = x_all["train"].shape[1]
            
            encoder = LogisticRegression(in_dim, num_classes)
            encoder = encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

            x = torch.cat(list(x_all.values()))
            y = torch.cat(list(y_all.values()))
            num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
            num_nodes = num_train + num_val + num_test
            train_mask = torch.arange(num_train, device=device)
            val_mask = torch.arange(num_train, num_train + num_val, device=device)
            test_mask = torch.arange(num_train + num_val, num_nodes, device=device)
            
            final_acc, estp_acc = linear_probing_for_inductive_node_classiifcation(encoder, x, y, (train_mask, val_mask, test_mask), optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc
    else:
        raise NotImplementedError


def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)# 
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            test_out = np.where(test_out >= 0, 1, 0)

            val_acc = f1_score(val_label, val_out, average="micro")
            test_acc = f1_score(test_label, test_out, average="micro")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    if mute:
        print(f"# IGNORE: --- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f},  Final-TestAcc: {test_acc:.4f}--- ")
    else:
        print(f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- ")

    return test_acc, best_val_test_acc


def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    train_loader, val_loader, test_loader, eval_train_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))

    if isinstance(train_loader, list) and len(train_loader) ==1:
        train_loader = [train_loader[0].to(device)]
        eval_train_loader = train_loader
    if isinstance(val_loader, list) and len(val_loader) == 1:
        val_loader = [val_loader[0].to(device)]
        test_loader = val_loader

    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for subgraph in train_loader:
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
        
        if epoch == (max_epoch//2):
            evaluete(model, (eval_train_loader, val_loader, test_loader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    if dataset_name == "ppi":
        dataset_name = "ppi-large"
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 

    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    (
        train_dataloader,
        valid_dataloader, 
        test_dataloader, 
        eval_train_dataloader, 
        num_features, 
        num_classes
    ) = load_inductive_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if not load_model:
            model = pretrain(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        model = model.cpu()

        model = model.to(device)
        model.eval()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc = evaluete(model, (eval_train_dataloader, valid_dataloader, test_dataloader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, es_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_f1: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_f1: {estp_acc:.4f}±{es_acc_std:.4f}")


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
