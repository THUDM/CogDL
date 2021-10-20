import torch
from torch.utils.data import DataLoader

from .. import DataWrapper
from cogdl.models.nn.sagn import prepare_labels, prepare_feats


class SAGNDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--label-nhop", type=int, default=3)
        parser.add_argument("--threshold", type=float, default=0.3)
        parser.add_argument("--nhop", type=int, default=3)
        # fmt: on

    def __init__(self, dataset, batch_size, label_nhop, threshold, nhop):
        super(SAGNDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_nhop = label_nhop
        self.nhop = nhop
        self.threshold = threshold

        self.label_emb, self.labels_with_pseudos, self.probs = None, None, None
        self.multihop_feats = None
        self.train_nid_with_pseudos = self.dataset.data.train_nid

        self.refresh_per_epoch("train")

    def train_wrapper(self):
        return DataLoader(self.train_nid_with_pseudos, batch_size=self.batch_size, shuffle=False)

    def val_wrapper(self):
        val_nid = self.dataset.data.val_nid
        return DataLoader(val_nid, batch_size=self.batch_size, shuffle=False)

    def test_wrapper(self):
        test_nid = self.dataset.data.test_nid
        return DataLoader(test_nid, batch_size=self.batch_size, shuffle=False)

    def post_stage_wrapper(self):
        data = self.dataset.data
        train_nid, val_nid, test_nid = data.train_nid, data.val_nid, data.test_nid
        all_nid = torch.cat([train_nid, val_nid, test_nid])
        return DataLoader(all_nid.numpy(), batch_size=self.batch_size, shuffle=False)

    def pre_stage_transform(self, batch):
        return self.train_transform(batch)

    def pre_transform(self):
        self.multihop_feats = prepare_feats(self.dataset, self.label_nhop)

    def train_transform(self, batch):
        batch_x = [x[batch] for x in self.multihop_feats]
        batch_x = torch.stack(batch_x)
        if self.label_emb is not None:
            batch_y_emb = self.label_emb[batch]
        else:
            batch_y_emb = None
        y = self.labels_with_pseudos[batch]
        return [batch_x, batch_y_emb, y]

    def val_transform(self, batch):
        batch_x = [x[batch] for x in self.multihop_feats]
        batch_x = torch.stack(batch_x)

        if self.label_emb is not None:
            batch_y_emb = self.label_emb[batch]
        else:
            batch_y_emb = None
        y = self.dataset.data.y[batch]
        return [batch_x, batch_y_emb, y]

    def test_transform(self, batch):
        return self.val_transform(batch)

    def pre_stage(self, stage, model_w_out):
        dataset = self.dataset
        probs = model_w_out
        with torch.no_grad():
            (label_emb, labels_with_pseudos, train_nid_with_pseudos) = prepare_labels(
                dataset, stage, self.label_nhop, self.threshold, probs=probs
            )
        self.label_emb = label_emb
        self.labels_with_pseudos = labels_with_pseudos
        self.train_nid_with_pseudos = train_nid_with_pseudos
