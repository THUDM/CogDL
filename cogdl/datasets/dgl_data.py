from dgl.data.tu import TUDataset

from . import register_dataset


@register_dataset("dgl-mutag")
class MUTAGDataset(TUDataset):
    def __init__(self):
        dataset = "MUTAG"
        super(MUTAGDataset, self).__init__(name=dataset)
        self.num_features = 0
        self.num_classes = self.num_labels[0]

@register_dataset("dgl-collab")
class CollabDataset(TUDataset):
    def __init__(self):
        dataset = "COLLAB"
        super(CollabDataset, self).__init__(name=dataset)
        self.num_features = 0
        self.num_classes = self.num_labels[0]

@register_dataset("dgl-imdb-b")
class ImdbBinaryDataset(TUDataset):
    def __init__(self):
        dataset = "IMDB-BINARY"
        super(ImdbBinaryDataset, self).__init__(name=dataset)
        self.num_features = 0
        self.num_classes = self.num_labels[0]

@register_dataset("dgl-imdb-m")
class ImdbMultiDataset(TUDataset):
    def __init__(self):
        dataset = "IMDB-MULTI"
        super(ImdbMultiDataset, self).__init__(name=dataset)
        self.num_features = 0
        self.num_classes = self.num_labels[0]

@register_dataset("dgl-proteins")
class ProtainsDataset(TUDataset):
    def __init__(self):
        dataset = "PROTEINS"
        super(ProtainsDataset, self).__init__(name=dataset)
        self.num_features = 0
        self.num_classes = self.num_labels[0]
