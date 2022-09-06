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
    "num_workers": 1,
    "unsup" : True,
    
    "pretrain": False,
    "freeze": False,
    "finetune": False,
    "devices":[0],
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

    #pretrain
    # dataset_name = 'gcc_academic'
    dataset_name = 'gcc_academic gcc_dblp_netrep gcc_dblp_snap gcc_facebook gcc_imdb gcc_livejournal'
    args = get_default_args_for_nc(dataset_name, method, mw="gcc_mw", dw="gcc_dw")
    args.pretrain = True
    args.no_test = True
    
    args.devices = [2]
    args.epochs = 1
    args.num_copies = 6
    args.num_workers = 12
    args.batch_size = 32
    args.num_samples = 2000
    train(args)
    
    #freeze
    # dataset_name = 'h-index'
    # args = get_default_args_for_nc(dataset_name, method, mw="gcc_mw", dw="gcc_dw")
    # args.epochs = 0
    # args.freeze = True
    # args.load_model_path = "./saved/already_trained_model/gcc_pretrain.pt"
    # train(args)


    # finetune 
    # dataset_name = 'h-index'
    # args = get_default_args_for_nc(dataset_name, method, mw="gcc_mw", dw="gcc_dw")
    # args.finetune = True
    # args.epochs  = 30
    # args.load_model_path = "./saved/already_trained_model/gcc_pretrain.pt"
    # train(args)
