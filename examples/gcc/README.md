# GCC
## Fork from CogDL and revise GCC
The original GCC source from https://github.com/THUDM/GCC.

## Introduction
The file run_gcc.py provides threee exmaples.
- pretrain (pretrain with six graph datasets)
- freeze (Generate node representations on downstream task graph utilizing pre-trained model parameters and test with the classifier)
- finetune (Fine-tuning on downstream task graph data with labels)

## Notice
1. ❗ ❗ Please check the version of numpy and numba.
- numpy == 1.17.3
- numba == 0.54.1
2. When executing the freeze operation, please make sure `epochs=0`.

## Pretrain
The dataset for pretrain is provided at https://pan.baidu.com/#bdlink=MzU2ODdmMjI2NjMzZGY2YWY2YWUwZjExMzAyMzQ1Y2EjZTFkZDEzNzNhNWM0YWU2ZmI2ZT \
Please put them on the `./data`

## Others
You can also use our already pretrained model parameters to test or fine-tune on different downstream task graphs. Just put the `gcc_pretrain.pt` into the file path `./saved/already_trained_model/`. This file is available for download on baidu netdisk. \
https://pan.baidu.com/s/1HkZ0H1ZeSBCBQlR3k3ekjg \
fkzo 