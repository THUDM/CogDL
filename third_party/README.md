# Third-party libraries

[dgNN](https://github.com/dgSPARSE/dgNN) is currently used for fast GAT training with much less GPU memory. 

[ActNN](https://github.com/ucbrise/actnn) can reduce the training memory footprint by compressing the saved activations.

[FastMoE](https://github.com/laekov/fastmoe) can be used for GNN models wtih the Mixture of Experts (MoE).

## Installation

For dgNN,
```bash
cd dgNN
python setup.py install
```

For ActNN,
```bash
cd actnn/actnn
pip install -v -e .
```

For FastMoE,
```bash
cd fastmoe
python setup.py install
```
