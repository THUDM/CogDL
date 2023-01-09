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
The dataset for pretrain is provided at this [link](https://pan.baidu.com/#bdlink=M2VmNDhlY2M5ZWM3NDA5YzhjMGUxODJhNTBiMmNkNDcjYjQ2MDZhZTNmZDkzNTI5YjMwMjNkOWI1MTNiMGFkMDcjMjM2NjE1NTkjZGF0YS5wdAo5YjVmOGU0NzE2ODZhNjBiYzIwNDQ2MzgxNjRiY2ZlZSM2MDNiOGU0ZDMwZTExNTQyY2NhZjM1YjY0ODc1MjliOCMyNDIwNzc4NzkjZGF0YS5wdAo0MzAyMjc4M2JkYzUzNjJmYjk0ODE2NzI5MWY4NzRmYyM5NTk0YzEwN2QzZTE0M2U2MjgwNDQ5NmJhZGM4NDc0NCMyNzQyMTI1MDQ3I2RhdGEucHQKOWM1YmJmZDAxOWIyMzc2NzAwMDhhOTBjMGM3YmE5ZjMjODIwNGIwMDU5MWY2ZmIwM2UyN2U1OWYwNDA0MGNlZjMjOTc1NzI3OTI3I2RhdGEucHQKYmNiNTljMTFiYzQ2NWRlMDQxOGZkNjM1ZTFjZGM1MTAjZTMxNjQ2ZGY1NDZjYmM3ODY3YTQ3ZGY0YjNhMTM4NWYjNjcxOTI2OTUjZGF0YS5wdAo2ZGQyYTcwZTc3YTgwOGUzNjlmM2IyNjdmZDM4NzNlMyM3ZTEzY2MyNGQ0YWViNmY2NzBmY2JmYTkzODBjYzY2MiMxNTE0NzE0NDg3I2RhdGEucHQKMGViZWQ2ZGIyMjkwOTI2MWM1M2ZhY2IxNzkxZDBjMTEjZjA3MDM3MDE4MmEyZDU3Njc5NTZiNGY3NjFiNTlhNWMjMTEwNjYzMDM2I2VkZ2VsaXN0LnR4dApiMTU1ZDAyZWIxMWUxODM0ZmExOWE1NGUzODYyYzc1MyM3YmE3ODY1ODY2OTljOTMzNmI2ZDM2N2RhNTJhZGQyMiMxMzQwODc5MTg4I2VkZ2VsaXN0LnR4dAo3Yzg1NzQ3YWVkNzZjM2NkYTU1ZWExNTY0NTE3MGM0ZiNmZDBlMTIwZjRlNDg0Y2I4Y2IyYzkyZWY0ZjFhNDE4NSM3MDM1OTM3MDgjZWRnZWxpc3QudHh0CmEwZTMxYzc0YzhlMDc3NTE1ZDE5MTk1NDRiZWNlZWFhI2NmOWY4OTM2YmVhOTczY2VkOGM0YmZkM2E5ZjA4ZjNhIzI5MzE5MTI2I2VkZ2VsaXN0LnR4dAo4ODIxMzdjZjNmMDFmNmMwODJhZDY3ZTM0MjcxMWVjYyM2OGI0MmYyMmM2MWU1YWI1OWE0ZDRiYTJmNDNiZjYwZSM5ODg2MzQ4I2VkZ2VsaXN0LnR4dAo4ZDViZTRkZDc2NmQ3M2MyNzliYWZhMmU2MjdjNGU3OCNmNDFjZDg5NTU4MzIxNzU1NzI3OGQ3NjFkOWE2ZDlkNyM0MzE3NjgwODYjZWRnZWxpc3QudHh0).

Please put them on the path `./data`

## Performance
Reference performance numbers for three datasets.
|          |Wiki-CS|Computers|Photo    |CS   |Physics| 
|------    |------ |---------|---------|-----|-------| 
|Paper     |79.98  |90.34    |93.17    |93.31|95.73  |
|Namkyeong |79.50  |88.21    |92.76    |92.49|94.89  |
|CogDL     |79.76  |87.99    |92.91    |93.05|92.74  |

## Others
You can also use our already pretrained model parameters to test or fine-tune on different downstream task graphs. Just put the `gcc_pretrain.pt` into the file path `./saved/already_trained_model/`. This file is available for download on baidu netdisk. \
https://pan.baidu.com/s/1HkZ0H1ZeSBCBQlR3k3ekjg \
fkzo 