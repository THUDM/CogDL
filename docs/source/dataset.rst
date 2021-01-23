Dataset
=========

In order to add a dataset into CogDL, you should know your dataset's format. We have provided several graph format like edgelist, matlab_matrix and pyg.
If the format of your dataset is the same as the `ppi` dataset, which contains two matrices: `network` and `group`, you can register your dataset directly use the following code.

.. code-block:: python

    @register_dataset("ppi")
    class PPIDataset(MatlabMatrix):
        def __init__(self):
            dataset, filename = "ppi", "Homo_sapiens"
            url = "http://snap.stanford.edu/node2vec/"
            path = osp.join("data", dataset)
            super(PPIDataset, self).__init__(path, filename, url)

You should declare the name of the dataset, the name of file and the url, where our script can download resource. More implemented datasets are at 
https://github.com/THUDM/cogdl/tree/master/cogdl/datasets.
