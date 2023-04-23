dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="mutag"
[ -z "${device}" ] && device=-1

python main_graph.py \
	--device $device \
	--dataset $dataset \
	--mask_rate 0.5 \
	--encoder "gin" \
	--decoder "gin" \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_hidden 512 \
	--num_heads 2 \
	--max_epoch 100 \
	--max_epoch_f 0 \
	--lr 0.00015 \
	--weight_decay 0.0 \
	--activation prelu \
	--optimizer adam \
	--drop_edge_rate 0.0 \
	--loss_fn "sce" \
	--seeds 0 1 2 3 4 \
	--linear_prob \
	--use_cfg \
