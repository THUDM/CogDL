Install
=======

- Python version >= 3.6
- PyTorch version >= 1.6.0
- PyTorch Geometric (recommended)
- Deep Graph Library (optional)

Please follow the instructions here to install PyTorch: https://github.com/pytorch/pytorch#installation, PyTorch Geometric https://github.com/rusty1s/pytorch_geometric/#installation and Deep Graph Library https://docs.dgl.ai/install/index.html.

Install cogdl with other dependencies: 

.. code-block:: python

    pip install cogdl


If you want to experiment with the latest CogDL features which did not get released yet, you can install CogDL via:

.. code-block:: python

    git clone git@github.com:THUDM/cogdl.git
    cd cogdl
    pip install -e .
