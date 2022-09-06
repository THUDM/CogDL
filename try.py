import wandb
import socket

from cogdl import experiment

# experiment(dataset="cora", model="gcn")


# wandb init
# wandb.init(config=None,
#            project="GCC",
#            entity="hwangyeong",
#            notes=socket.gethostname(),
#            name="CogDL_gcc",
#            dir='results/wandb_res',
#            job_type="training",
#            reinit=True)

experiment(dataset="gcc_academic gcc_dblp_netrep gcc_dblp_snap gcc_facebook gcc_imdb gcc_livejournal", 
           model="gcc",
           lr=0.005,
           weight_decay=1e-05,
           clip_grad_norm=1.,
           beta1=0.9,
           beta2=0.999,
           n_warmup_steps=0.1,
           devices=[3],
           epochs=1,
           pretrain=True,
           no_test=True,
           unsup=True, #must
           do_valid=False,
           do_test=False
           )

#freeze
# experiment(dataset="h-index", 
#            model="gcc",
#            epochs=0, #must
#            freeze=True, #must
#            load_model_path="./saved/already_trained_model/gcc_pretrain.pt",
#         #    load_model_path="./saved/Pretrain_academic_dblp-netrep_dblp-snap_facebook_imdb_livejournal_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_rwr_ft_False_deg_16_pos_32_momentum_0.999/gcc_pretrain.pt"
#            )

#finetune
# experiment(dataset="usa-airport", 
#            model="gcc",
#            epochs=30, #must
#            finetune=True, #must
#         #    load_model_path="./saved/Pretrain_academic_dblp-netrep_dblp-snap_facebook_imdb_livejournal_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_rwr_ft_False_deg_16_pos_32_momentum_0.999/gcc_pretrain.pt"
#            load_model_path="./saved/already_trained_model/gcc_pretrain.pt"
#            )