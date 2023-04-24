dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="ogbn-arxiv"
[ -z "${device}" ] && device=0

CUDA_VISIBLE_DEVICES=$device \
	python main_large.py \
	--device 0 \
	--dataset $dataset \
	--mask_type "mask" \
	--mask_rate 0.5 \
	--remask_rate 0.5 \
	--num_remasking 3 \
	--in_drop 0.2 \
	--attn_drop 0.2 \
	--num_layers 4 \
	--num_dec_layers 1 \
	--num_hidden 1024 \
	--num_heads 4 \
	--num_out_heads 1 \
	--encoder "gat" \
	--decoder "gat" \
	--max_epoch 60 \
	--max_epoch_f 1000 \
	--lr 0.002 \
	--weight_decay 0.04 \
	--lr_f 0.005 \
	--weight_decay_f 1e-4 \
	--activation "prelu" \
	--optimizer "adamw" \
	--drop_edge_rate 0.5 \
	--loss_fn "sce" \
	--alpha_l 4 \
	--mask_method "random" \
	--scheduler \
    --batch_size 512 \
	--batch_size_f 256 \
	--seeds 0 \
	--residual \
	--norm "layernorm" \
	--sampling_method "lc" \
	--label_rate 1.0 \
	--lam 1.0 \
	--momentum 0.996 \
	--linear_prob \
	--use_cfg \
	--ego_graph_file_path "./lc_ego_graphs/${dataset}-lc-ego-graphs-256.pt" \
	--data_dir "./dataset" \
	# --logging
