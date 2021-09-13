import re

import torch
from cogdl.data import Graph, Adjacency


def batch_graphs(graphs):
    return Batch.from_data_list(graphs, class_type=Graph)


class Batch(Graph):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`cogdl.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)
        self.batch = batch
        self.__data_class__ = Graph
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, class_type=None):
        r"""Constructs a batch object from a python list holding
        :class:`cogdl.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        # keys = [set(data.keys) for data in data_list]
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        if class_type is not None:
            batch = class_type()
        else:
            batch = Batch()
            batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        # for key in follow_batch:
        #     batch["{}_batch".format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        num_nodes_cum = [0]
        num_edges_cum = [0]
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.cat_dim(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                # if key in follow_batch:
                #     item = torch.full((size,), i, dtype=torch.long)
                #     batch["{}_batch".format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                num_nodes_cum.append(num_nodes + num_nodes_cum[-1])
                num_edges_cum.append(data.num_edges + num_edges_cum[-1])
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)
        if num_nodes is None:
            batch.batch = None
        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            elif isinstance(item, Adjacency):
                target = Adjacency()
                for k in item.keys:
                    if item[k] is None:
                        continue
                    if k == "row" or k == "col":
                        _item = torch.cat(
                            [x[k] + num_nodes_cum[i] for i, x in enumerate(batch[key])], dim=item.cat_dim(k, None)
                        )
                    elif k == "row_ptr":
                        _item = torch.cat(
                            [x[k][:-1] + num_edges_cum[i] for i, x in enumerate(batch[key][:-1])],
                            dim=item.cat_dim(k, None),
                        )
                        _item = torch.cat([_item, batch[key][-1][k] + num_edges_cum[-2]], dim=item.cat_dim(k, None))
                    else:
                        _item = torch.cat([x[k] for i, x in enumerate(batch[key])], dim=item.cat_dim(k, None))
                    target[k] = _item
                batch[key] = target.to(item.device)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return bool(re.search("(index|face)", key))

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
