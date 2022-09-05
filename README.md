![CogDL](./docs/source/_static/cogdl-logo.png)
===

## fork from CogDL and revise GCC

在根目录下的 try_gcc.py 文件中运行gcc，其中配置了三种参数
- pretrain 预训练
- freeze 利用预训练参数在下游任务上测试
- finetune 利用预训练参数在下游任务上微调

### 导致bug的点
- 增加了每个epoch都存储参数功能，该功能可有可无，但需要适配其他方法。 \
cogdl/tainer/trainer.py 文件中第418到421行增加了每个epoch都执行post_stage操作,并且
修改了model_w.post_stage的参数，增加参数epoch。当epoch为-1是表示训练完成后，执行post_stage操作；当epoch为自然数时，表示在第几次epoch时执行model_w.post_stage操作。
- pretrain结束后无需valid和test操作，因此在raw_experiment()函数中因无返回结果，导致output_results(results_dict, tablefmt)出错，给问题尚未解决

### 注意：
1. ❗ ❗ 请确保numpy与numba版本如下
- numpy == 1.17.3
- numba == 0.54.1

2. 当执行freeze操作时，请确保参数epochs=0 （可以增加参数检查模块，但不知道放在哪）
3. 我们提供已经训练好的参数模型，可直接用于freeze及finetune，路径：/home/huanjing/work_document/NE/cogdl_gcc_revision/saved/already_trained_model/gcc_pretrain.pt
4. 项目数据位于服务器 /home/huanjing/work_document/NE/cogdl_gcc_revision/data 文件下