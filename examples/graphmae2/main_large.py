import logging
import os
import numpy as np
from tqdm import tqdm

import torch

from utils import (
    WandbLogger,
    build_args,
    create_optimizer,
    set_random_seed,
    load_best_configs,
    show_occupied_memory,
)
from models import build_model
from datasets.lc_sampler import (
    setup_training_dataloder, 
    setup_training_data,
)
from models.finetune import linear_probing_minibatch, finetune

import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def evaluate(
    model, 
    graph, feats, labels, 
    split_idx, 
    lr_f, weight_decay_f, max_epoch_f, 
    linear_prob=True, 
    device=0, 
    batch_size=256, 
    logger=None, ego_graph_nodes=None, 
    label_rate=1.0,
    full_graph_forward=False,
    shuffle=True,
):
    logging.info("Using `lc` for evaluation...")
    num_train, num_val, num_test = [split_idx[k].shape[0] for k in ["train", "valid", "test"]]
    print(num_train,num_val,num_test)

    train_g_idx = np.arange(0, num_train)
    val_g_idx = np.arange(num_train, num_train+num_val)
    test_g_idx = np.arange(num_train+num_val, num_train+num_val+num_test)

    train_ego_graph_nodes = [ego_graph_nodes[i] for i in train_g_idx]
    val_ego_graph_nodes = [ego_graph_nodes[i] for i in val_g_idx]
    test_ego_graph_nodes = [ego_graph_nodes[i] for i in test_g_idx]

    train_lbls, val_lbls, test_lbls = labels[train_g_idx], labels[val_g_idx], labels[test_g_idx]

    # labels = [train_lbls, val_lbls, test_lbls]
    assert len(train_ego_graph_nodes) == len(train_lbls)
    assert len(val_ego_graph_nodes) == len(val_lbls)
    assert len(test_ego_graph_nodes) == len(test_lbls)

    print(f"num_train: {num_train}, num_val: {num_val}, num_test: {num_test}")
    logging.info(f"-- train_ego_nodes:{len(train_ego_graph_nodes)}, val_ego_nodes:{len(val_ego_graph_nodes)}, test_ego_nodes:{len(test_ego_graph_nodes)} ---")
    

    if linear_prob:
        result = linear_probing_minibatch(model, graph, feats, [train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes], [train_lbls, val_lbls, test_lbls], lr_f=lr_f, weight_decay_f=weight_decay_f, max_epoch_f=max_epoch_f, batch_size=batch_size, device=device, shuffle=shuffle)
    else:
        max_epoch_f = max_epoch_f // 2

        if label_rate < 1.0:
            rand_idx = np.arange(len(train_ego_graph_nodes))
            np.random.shuffle(rand_idx)
            rand_idx = rand_idx[:int(label_rate * len(train_ego_graph_nodes))]
            train_ego_graph_nodes = [train_ego_graph_nodes[i] for i in rand_idx]
            train_lbls = train_lbls[rand_idx]

        logging.info(f"-- train_ego_nodes:{len(train_ego_graph_nodes)}, val_ego_nodes:{len(val_ego_graph_nodes)}, test_ego_nodes:{len(test_ego_graph_nodes)} ---")

        # train_lbls = (all_train_lbls, train_lbls)
        result = finetune(
            model, graph, feats, 
            [train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes], 
            [train_lbls, val_lbls, test_lbls], 
            split_idx=split_idx,
            lr_f=lr_f, weight_decay_f=weight_decay_f, max_epoch_f=max_epoch_f, use_scheduler=True, batch_size=batch_size, device=device, logger=logger, full_graph_forward=full_graph_forward,
        )
    return result


def pretrain(model, feats, graph, ego_graph_nodes, max_epoch, device, use_scheduler, lr, weight_decay, batch_size=512, sampling_method="lc", optimizer="adam", drop_edge_rate=0):
    logging.info("start training..")

    model = model.to(device)
    optimizer = create_optimizer(optimizer, model, lr, weight_decay)    

    dataloader = setup_training_dataloder(
        sampling_method, ego_graph_nodes, graph, feats, batch_size=batch_size, drop_edge_rate=drop_edge_rate)

    logging.info(f"After creating dataloader: Memory: {show_occupied_memory():.2f} MB")
    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    for epoch in range(max_epoch):
        epoch_iter = tqdm(dataloader)
        losses = []
        # assert (graph.in_degrees() > 0).all(), "after loading"

        for batch_g in epoch_iter:
            model.train()
            if drop_edge_rate > 0:
                batch_g, targets, _, node_idx, drop_g1, drop_g2 = batch_g
                batch_g = batch_g.to(device)
                drop_g1 = drop_g1.to(device)
                drop_g2 = drop_g2.to(device)
                x = batch_g.x
                loss = model(batch_g, x, targets, epoch, drop_g1, drop_g2)
            else:
                batch_g, targets, _, node_idx = batch_g
                batch_g = batch_g.to(device)
                x = batch_g.x
                loss = model(batch_g, x, targets, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            epoch_iter.set_description(f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            losses.append(loss.item())
            
        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

        print(f"# Epoch {epoch} | train_loss: {np.mean(losses):.4f}, Memory: {show_occupied_memory():.2f} MB")

    return model


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args)

    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    encoder = args.encoder
    decoder = args.decoder
    num_hidden = args.num_hidden
    drop_edge_rate = args.drop_edge_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    no_pretrain = args.no_pretrain
    logs = args.logging
    use_scheduler = args.scheduler
    batch_size = args.batch_size
    batch_size_f = args.batch_size_f
    sampling_method = args.sampling_method
    ego_graph_file_path = args.ego_graph_file_path
    data_dir = args.data_dir

    n_procs = torch.cuda.device_count()
    optimizer_type = args.optimizer
    label_rate = args.label_rate
    lam = args.lam
    full_graph_forward = hasattr(args, "full_graph_forward") and args.full_graph_forward and not linear_prob

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    set_random_seed(0)
    print(args)
    
    logging.info(f"Before loading data, occupied memory: {show_occupied_memory():.2f} MB") # in MB 
    feats, graph, labels, split_idx, ego_graph_nodes = setup_training_data(dataset_name, data_dir, ego_graph_file_path)
    if dataset_name == "ogbn-papers100M":
        pretrain_ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2] + ego_graph_nodes[3]
    else:
        pretrain_ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
    ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2] # * merge train/val/test = all

    logging.info(f"After loading data, occupied memory: {show_occupied_memory():.2f} MB") # in MB 

    args.num_features = feats.shape[1]

    if logs:
        logger = WandbLogger(log_path=f"{dataset_name}_loss_{loss_fn}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}", project="GraphMAE2", args=args)
    else:
        logger = None
    model_name = f"{encoder}_{decoder}_{num_hidden}_{num_layers}_{dataset_name}_{args.mask_rate}_{num_hidden}_checkpoint.pt"

    model = build_model(args)

    if not args.no_pretrain:
        # ------------- pretraining starts ----------------
        if not load_model:
            logging.info("---- start pretraining ----")
            model = pretrain(model, feats, graph, pretrain_ego_graph_nodes, max_epoch=max_epoch, device=device, use_scheduler=use_scheduler, lr=lr, 
            weight_decay=weight_decay, batch_size=batch_size, drop_edge_rate=drop_edge_rate,  
            sampling_method=sampling_method, optimizer=optimizer_type)
        
            model = model.cpu()
            logging.info(f"saving model to {model_dir}/{model_name}...")
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        # ------------- pretraining ends ----------------   

        if load_model:
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path)))
            logging.info(f"Loading Model from {args.checkpoint_path}...")
    else:
        logging.info("--- no pretrain ---")

    model = model.to(device)
    model.eval()

    logging.info("---- start finetuning / evaluation ----")

    final_accs = []
    for i,_ in enumerate(seeds):
        print(f"####### Run seed {seeds[i]}")
        set_random_seed(seeds[i])
        eval_model = build_model(args)
        eval_model.load_state_dict(model.state_dict())
        eval_model.to(device)

        print(f"features size : {feats.shape[1]}")
        logging.info("start evaluation...")
        final_acc = evaluate(
            eval_model, graph, feats, labels, split_idx,
            lr_f, weight_decay_f, max_epoch_f, 
            device=device, 
            batch_size=batch_size_f, 
            ego_graph_nodes=ego_graph_nodes, 
            linear_prob=linear_prob,
            label_rate=label_rate,
            full_graph_forward=full_graph_forward,
            shuffle=False if dataset_name == "ogbn-papers100M" else True
        )

        final_accs.append(float(final_acc))

        print(f"Run {seeds[i]} | TestAcc: {final_acc:.4f}")
        
    print(f"# final_acc: {np.mean(final_accs):.4f}, std: {np.std(final_accs):.4f}")
    
    if logger is not None:
        logger.finish()
