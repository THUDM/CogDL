# GCC
## Fork from CogDL and revise GCC
The original GCC source from https://github.com/THUDM/GCC.

## Introduction
The file run_gcc.py provides threee exmaples.
- pretrain (pretrain with six graph datasets)
- freeze (Generate node representations on downstream task graph utilizing pre-trained model parameters and test with the classifier)
- finetune (Fine-tuning on downstream task graph data with labels)

## Notice
1. The restart random walk is a time-consuming step, we use numba to achieve parallelism to optimize this process. Please check the version of numpy and numba. ❗ ❗ 
- numpy == 1.17.3
- numba == 0.54.1 \
Non-parallel restart random walks are also allowed, you just need to ensure that any version of numpy and numba and set the parameter `parallel=False`. (NOT RECOMMENDED)
2. When executing the freeze operation, please make sure `epochs=0`.

## Pretrain
The dataset for pretrain is provided at this [link](https://pan.baidu.com/s/13hVPt8KmOkdDQ_Wf45T4pg?pwd=y2j5).

Please put them on the path `./data`

## Performance
The results of this implementation is shown in the Table.
|      |usa-airport <br> (freeze)|h-index <br> (freeze)|usa-airport <br> (finetune)|h-index <br> (finetune)|
|------|:---:|:---:|:---:|:---:|
|paper |65.6 |75.2 |67.2 |80.6 |
|CogDL |63.8 |78.4 |69.8 |80.9 |


## Others
You can also use our already pretrained model parameters to test or fine-tune on different downstream task graphs. Just put the `gcc_pretrain.pt` into the file path `./saved/already_trained_model/`. This file is available for download on baidu netdisk. \
https://pan.baidu.com/s/1HkZ0H1ZeSBCBQlR3k3ekjg \
fkzo 