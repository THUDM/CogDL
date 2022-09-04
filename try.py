import wandb
import socket

from cogdl import experiment

# experiment(dataset="cora", model="gcn")


# wandb init
wandb.init(config=None,
           project="GCC",
           entity="hwangyeong",
           notes=socket.gethostname(),
           name="CogDL_gcc",
           dir='results/wandb_res',
           job_type="training",
           reinit=True)

experiment(dataset="gcc_academic gcc_dblp_netrep gcc_dblp_snap gcc_facebook gcc_imdb gcc_livejournal", 
           model="gcc",
           lr=0.005,
           weight_decay=1e-05,
           clip_grad_norm=1.,
           beta1=0.9,
           beta2=0.999,
           n_warmup_steps=0.1,
           devices=[3],
           epochs=100,
           pretrain=True,
           no_test=True,
           unsup=True, #must
           )