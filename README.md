![CogDL](./docs/source/_static/cogdl-logo.png)
===

## fork form CogDL

在根目录下的try_gcc文件中运行gcc，其中配置了三种参数
- pretrain 预训练
- freeze 利用预训练参数在下游任务上测试
- finetune 利用预训练参数在下游任务上微调

### 可能导致bug的点
- cogdl/tainer/trainer.py 文件中第418到421行增加了每个epoch都执行post_stage操作,并且
修改了model_w.post_stage的参数，增加参数epoch。当epoch为-1是表示训练完成后，执行post_stage操作；当epoch为自然数时，表示在第几次epoch时执行model_w.post_stage操作。

### 注意：
1. ❗ ❗ 请确保numpy与numba版本如下
- numpy == 1.17.3
- numba == 0.54.1

2. 当执行freeze操作时，请确保参数epochs=0
3. 我们提供已经训练好的参数模型，可直接用于freeze及finetune，路径：./saved/already_trained_model/gcc_pretrain.pt
