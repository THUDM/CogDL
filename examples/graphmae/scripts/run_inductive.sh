dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="ppi"
[ -z "${device}" ] && device=-1


python main_inductive.py \
	--device $device \
	--dataset $dataset \
	--mask_rate 0.5 \
	--encoder "gat" \
	--decoder "gat" \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 3 \
	--num_hidden 2048 \
	--num_heads 4 \
	--max_epoch 1000 \
	--max_epoch_f 500 \
	--lr 0.001 \
	--weight_decay 0 \
	--lr_f 0.005 \
	--weight_decay_f 0 \
	--activation prelu \
	--optimizer adam \
	--drop_edge_rate 0.0 \
	--loss_fn "sce" \
	--seeds 0 1 2 3 4 \
	--replace_rate 0.0 \
	--alpha_l 3 \
	--linear_prob \
	--use_cfg \
