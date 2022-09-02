# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 22:57:47 2022

@author: Yangshan

"""
# import wandb
import socket
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'+'/'+'..'))

import torch
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
from cogdl.datasets import build_dataset_from_name
from cogdl.options import get_default_args
from cogdl.wrappers.data_wrapper.pretraining.gcc_dw import GCCDataWrapper
from cogdl.wrappers.model_wrapper.pretraining.gcc_mw import GCCModelWrapper
from cogdl.datasets.gcc_data import USAAirportDataset
from cogdl.experiments import train
# from cogdl.experiments.

cuda_available = torch.cuda.is_available()
default_dict = {
    "dropout": 0.5,
    "patience": 2,
    "epochs": 3,
    "sampler": "none",
    "cpu": not cuda_available,
    "checkpoint": False,
    "auxiliary_task": "none",
    "eval_step": 1,
    "activation": "relu",
    "residual": False,
    "norm": None,
    "num_workers": 1,
    "unsup" : True,
    
    "pretrain": False,
    "freeze": False,
    "finetune": False,
    "hidden_size": 64,
    "output_size": 64,
    "positional_embedding_size":32,
    "degree_embedding_size": 16,
    "gnn_model": "gin",
    "num_layers": 5,
    "aug": "rwr",
    "rw_hops": 256,
    "num_samples": 2000,
    "nce_k": 16384,
    "nce_t": 0.07,
    "norm": True,
    "momentum": 0.999,
    "lr": 0.005,
    "weight_decay": 1e-05,
    "beta1": 0.9,
    "beta2": 0.999,
    "clip_grad_norm": 1.,
    "norm": True,
    "n_warmup_steps": 0.1,
    "save_model_path":"saved"
}
# Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999

def get_default_args_for_nc(dataset, model, dw="node_classification_dw", mw="node_classification_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args

# dataset = build_dataset_from_name('cora')[0]
if __name__ == "__main__":
    method = 'gcc'
    # dataset_name = "gcc_dblp_snap"
    # dataset_name = 'gcc_academic gcc_dblp_netrep gcc_dblp_snap gcc_facebook gcc_imdb gcc_livejournal'
    # dataset_name = 'usa-airport h-index'
    dataset_name = 'usa-airport'
    # dataset = USAAirportDataset()

    args = get_default_args_for_nc(dataset_name, method, mw="gcc_mw", dw="gcc_dw")
    # args = get_default_args_for_nc("cora", "gcc", mw="gcc_mw", dw="gcc_dw")
    args.devices = [1]
    args.epochs = 100
    args.num_copies = 6
    args.num_workers = 12
    args.batch_size = 32
    
    # args.pretrain = True
    # # args.degree_input = False
    # args.num_samples = 2000
    # args.no_test = True

    # wandb init
    # wandb.init(config=args,
    #            project="GCC",
    #            entity="hwangyeong",
    #            notes=socket.gethostname(),
    #            name="Cogdl_GCC",
    #            dir='results/wandb_res',
    #            job_type="training",
    #            reinit=True)

    # train(args)

    #freeze   
    args.epochs = 0
    args.freeze = True
    args.load_model_path = "./saved/already_trained_model/gcc_pretrain.pt"
    train(args)


    # finetune 
    # args.finetune = True
    # args.epochs  = 30
    # args.load_model_path = "./saved/Pretrain_academic_dblp-netrep_dblp-snap_facebook_imdb_livejournal_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_rwr_ft_False_deg_16_pos_32_momentum_0.999/gcc_pretrain.pt"
    # train(args)
