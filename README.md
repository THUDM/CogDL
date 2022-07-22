![CogDL](./docs/source/_static/cogdl-logo.png)
===

[![PyPI Latest Release](https://badge.fury.io/py/cogdl.svg)](https://pypi.org/project/cogdl/)
[![Build Status](https://travis-ci.org/THUDM/cogdl.svg?branch=master)](https://travis-ci.org/THUDM/cogdl)
[![Documentation Status](https://readthedocs.org/projects/cogdl/badge/?version=latest)](https://cogdl.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/cogdl)](https://pepy.tech/project/cogdl)
[![Coverage Status](https://coveralls.io/repos/github/THUDM/cogdl/badge.svg?branch=master)](https://coveralls.io/github/THUDM/cogdl?branch=master)
[![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/THUDM/cogdl/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**[Homepage](https://cogdl.ai)** | **[Paper](https://arxiv.org/abs/2103.00959)** | **[Documentation](https://cogdl.readthedocs.io)** | **[Discussion Forum](https://discuss.cogdl.ai)** | **[Dataset](./cogdl/datasets/README.md)** | **[中文](./README_CN.md)**

CogDL is a graph deep learning toolkit that allows researchers and developers to easily train and compare baseline or customized models for node classification, graph classification, and other important tasks in the graph domain. 

We summarize the contributions of CogDL as follows:

- **Efficiency**: CogDL utilizes well-optimized operators to speed up training and save GPU memory of GNN models.
- **Ease of Use**: CogDL provides easy-to-use APIs for running experiments with the given models and datasets using hyper-parameter search.
- **Extensibility**: The design of CogDL makes it easy to apply GNN models to new scenarios based on our framework.

## ❗ News

- A free GNN course provided by CogDL Team is present at [this link](https://cogdl.ai/gnn2022/). We also provide a [discussion forum](https://discuss.cogdl.ai) for Chinese users. 

- The new **v0.5.3 release** supports mixed-precision training by setting \textit{fp16=True} and provides a basic [example](https://github.com/THUDM/cogdl/blob/master/examples/jittor/gcn.py) written by [Jittor](https://github.com/Jittor/jittor). It also updates the tutorial in the document, fixes downloading links of some datasets, and fixes potential bugs of operators. 

- The new **v0.5.2 release** adds a GNN example for ogbn-products and updates geom datasets. It also fixes some potential bugs including setting devices, using cpu for inference, etc.

- The new **v0.5.1 release** adds fast operators including SpMM (cpu version) and scatter_max (cuda version). It also adds lots of datasets for node classification which can be found in [this link](./cogdl/datasets/rd2cd_data.py). 🎉

<details>
<summary>
News History
</summary>
<br/>

- The new **v0.5.0 release** designs and implements a unified training loop for GNN. It introduces `DataWrapper` to help prepare the training/validation/test data and `ModelWrapper` to define the training/validation/test steps. 🎉

- The new **v0.4.1 release** adds the implementation of Deep GNNs and the recommendation task. It also supports new pipelines for generating embeddings and recommendation. Welcome to join our tutorial on KDD 2021 at 10:30 am - 12:00 am, Aug. 14th (Singapore Time). More details can be found in https://kdd2021graph.github.io/. 🎉

- The new **v0.4.0 release** refactors the data storage (from `Data` to `Graph`) and provides more fast operators to speed up GNN training. It also includes many self-supervised learning methods on graphs. BTW, we are glad to announce that we will give a tutorial on KDD 2021 in August. Please see [this link](https://kdd2021graph.github.io/) for more details. 🎉

- CogDL supports GNN models with Mixture of Experts (MoE). You can install [FastMoE](https://github.com/laekov/fastmoe) and try **[MoE GCN](./cogdl/models/nn/moe_gcn.py)** in CogDL now!

- The new **v0.3.0 release** provides a fast spmm operator to speed up GNN training. We also release the first version of **[CogDL paper](https://arxiv.org/abs/2103.00959)** in arXiv. You can join [our slack](https://join.slack.com/t/cogdl/shared_invite/zt-b9b4a49j-2aMB035qZKxvjV4vqf0hEg) for discussion. 🎉🎉🎉

- The new **v0.2.0 release** includes easy-to-use `experiment` and `pipeline` APIs for all experiments and applications. The `experiment` API supports automl features of searching hyper-parameters. This release also provides `OAGBert` API for model inference (`OAGBert` is trained on large-scale academic corpus by our lab). Some features and models are added by the open source community (thanks to all the contributors 🎉).

- The new **v0.1.2 release** includes a pre-training task, many examples, OGB datasets, some knowledge graph embedding methods, and some graph neural network models. The coverage of CogDL is increased to 80%. Some new APIs, such as `Trainer` and `Sampler`, are developed and being tested. 

- The new **v0.1.1 release** includes the knowledge link prediction task, many state-of-the-art models, and `optuna` support. We also have a [Chinese WeChat post](https://mp.weixin.qq.com/s/IUh-ctQwtSXGvdTij5eDDg) about the CogDL release.

</details>

## Getting Started

### Requirements and Installation

- Python version >= 3.7
- PyTorch version >= 1.7.1

Please follow the instructions here to install PyTorch (https://github.com/pytorch/pytorch#installation).

When PyTorch has been installed, cogdl can be installed using pip as follows:

```bash
pip install cogdl
```

Install from source via:

```bash
pip install git+https://github.com/thudm/cogdl.git
```

Or clone the repository and install with the following commands:

```bash
git clone git@github.com:THUDM/cogdl.git
cd cogdl
pip install -e .
```

## Usage

### API Usage

You can run all kinds of experiments through CogDL APIs, especially `experiment`. You can also use your own datasets and models for experiments. 
A quickstart example can be found in the [quick_start.py](https://github.com/THUDM/cogdl/tree/master/examples/quick_start.py). More examples are provided in the [examples/](https://github.com/THUDM/cogdl/tree/master/examples/).

```python
from cogdl import experiment

# basic usage
experiment(dataset="cora", model="gcn")

# set other hyper-parameters
experiment(dataset="cora", model="gcn", hidden_size=32, epochs=200)

# run over multiple models on different seeds
experiment(dataset="cora", model=["gcn", "gat"], seed=[1, 2])

# automl usage
def search_space(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_uniform("dropout", 0.5, 0.8),
    }

experiment(dataset="cora", model="gcn", seed=[1, 2], search_space=search_space)
```

### Command-Line Usage

You can also use `python scripts/train.py --dataset example_dataset --model example_model` to run example_model on example_data.

- --dataset, dataset name to run, can be a list of datasets with space like `cora citeseer`. Supported datasets include
'cora', 'citeseer', 'pumbed', 'ppi', 'wikipedia', 'blogcatalog', 'flickr'. More datasets can be found in the [cogdl/datasets](https://github.com/THUDM/cogdl/tree/master/cogdl/datasets).
- --model, model name to run, can be a list of models like `gcn gat`. Supported models include
'gcn', 'gat', 'graphsage', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'. More models can be found in the [cogdl/models](https://github.com/THUDM/cogdl/tree/master/cogdl/models).

For example, if you want to run GCN and GAT on the Cora dataset, with 5 different seeds:

```bash
python scripts/train.py --dataset cora --model gcn gat --seed 0 1 2 3 4
```

Expected output:

| Variant          | test_acc       | val_acc        |
|------------------|----------------|----------------|
| ('cora', 'gcn')  | 0.8050±0.0047  | 0.7940±0.0063  |
| ('cora', 'gat')  | 0.8234±0.0042  | 0.8088±0.0016  |

If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.


## ❗ FAQ

<details>
<summary>
How to contribute to CogDL?
</summary>
<br/>

If you have a well-performed algorithm and are willing to implement it in our toolkit to help more people, you can first [open an issue](https://github.com/THUDM/cogdl/issues) and then create a pull request, detailed information can be found [here](https://help.github.com/en/articles/creating-a-pull-request). 

Before committing your modification, please first run `pre-commit install` to setup the git hook for checking code format and style using `black` and `flake8`. Then the `pre-commit` will run automatically on `git commit`! Detailed information of `pre-commit` can be found [here](https://pre-commit.com/).
</details>

<details>
<summary>
How to enable fast GNN training?
</summary>
<br/>
CogDL provides a fast sparse matrix-matrix multiplication operator called [GE-SpMM](https://arxiv.org/abs/2007.03179) to speed up training of GNN models on the GPU. 
The feature will be automatically used if it is available.
Note that this feature is still in testing and may not work under some versions of CUDA.
</details>

<details>
<summary>
How to run parallel experiments with GPUs on several models?
</summary>
<br/>

If you want to run parallel experiments on your server with multiple GPUs on multiple models, GCN and GAT, on the Cora dataset:

```bash
$ python scripts/train.py --dataset cora --model gcn gat --hidden-size 64 --devices 0 1 --seed 0 1 2 3 4
```

Expected output:

| Variant         | Acc           |
| --------------- | ------------- |
| ('cora', 'gcn') | 0.8236±0.0033 |
| ('cora', 'gat') | 0.8262±0.0032 |
</details>

<details>
<summary>
How to use models from other libraries?
</summary>
<br/>
If you are familiar with other popular graph libraries, you can implement your own model in CogDL using modules from PyTorch Geometric (PyG).
For the installation of PyG, you can follow the instructions from PyG (https://github.com/rusty1s/pytorch_geometric/#installation).
For the quick-start usage of how to use layers of PyG, you can find some examples in the [examples/pyg](https://github.com/THUDM/cogdl/tree/master/examples/pyg/).
</details>

<details>
<summary>
How to make a successful pull request with unit test
</summary>
<br/>
To have a successful pull request, you need to have at least (1) your model implementation and (2) a unit test.

You might be confused why your pull request was rejected because of 'Coverage decreased ...' issue even though your model is working fine locally. This is because you have not included a unit test, which essentially runs through the extra lines of code you added. The Travis CI service used by Github conducts all unit tests on the code you committed and checks how many lines of the code have been checked by the unit tests, and if a significant portion of your code has not been checked (insufficient coverage), the pull request is rejected.

So how do you do a unit test? 

* Let's say you implement a GNN model in a script `models/nn/abcgnn.py` that does the task of node classification. Then, you need to add a unit test inside the script `tests/tasks/test_node_classification.py` (or whatever relevant task your model does). 
* To add the unit test, you simply add a function *test_abcgnn_cora()* (just follow the format of the other unit tests already in the script), fill it with required arguments and the last line in the function *'assert 0 <= ret["Acc"] <= 1'* is the very basic sanity check conducted by the unit test. 
* After modifying `tests/tasks/test_node_classification.py`, commit it together with your `models/nn/abcgnn.py` and your pull request should pass.
</details>

## CogDL Team
CogDL is developed and maintained by [Tsinghua, ZJU, BAAI, DAMO Academy, and ZHIPU.AI](https://cogdl.ai/about/). 

The core development team can be reached at [cogdlteam@gmail.com](mailto:cogdlteam@gmail.com).

## Citing CogDL

Please cite [our paper](https://arxiv.org/abs/2103.00959) if you find our code or results useful for your research:

```
@article{cen2021cogdl,
    title={CogDL: A Toolkit for Deep Learning on Graphs},
    author={Yukuo Cen and Zhenyu Hou and Yan Wang and Qibin Chen and Yizhen Luo and Zhongming Yu and Hengrui Zhang and Xingcheng Yao and Aohan Zeng and Shiguang Guo and Yuxiao Dong and Yang Yang and Peng Zhang and Guohao Dai and Yu Wang and Chang Zhou and Hongxia Yang and Jie Tang},
    journal={arXiv preprint arXiv:2103.00959},
    year={2021}
}
```
