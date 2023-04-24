dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device=0

CUDA_VISIBLE_DEVICES=$device \
	python main_full_batch.py \
	--device 0 \
	--dataset $dataset \
	--mask_method "random" \
    --remask_method "fixed" \
	--mask_rate 0.5 \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_dec_layers 1 \
	--num_hidden 256 \
	--num_heads 4 \
	--num_out_heads 1 \
	--encoder "gat" \
	--decoder "gat" \
	--max_epoch 1000 \
	--max_epoch_f 300 \
	--lr 0.001 \
	--weight_decay 0.04 \
	--lr_f 0.005 \
	--weight_decay_f 1e-4 \
	--activation "prelu" \
	--loss_fn "sce" \
	--alpha_l 3 \
	--scheduler \
	--seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
	--lam 0.5 \
	--linear_prob \
	--data_dir "./dataset" \
    --use_cfg
