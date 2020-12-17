import torch
from torch import nn

from tqdm import tqdm

from .base_trainer import BaseTrainer


class SelfSupervisedTrainer(BaseTrainer):
    def __init__(self, args):
        super(SelfSupervisedTrainer, self).__init__()
        self.device = args.device_id[0]
        self.epochs = args.max_epoch
        self.patience = args.patience

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def fit(self, model, data):
        best = 1e9
        cnt_wait = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

        epoch_iter = tqdm(range(self.epochs))

        features = data.x
        edge_index = data.edge_index
        # edge_weight = data.edge_index if hasattr(data, "edge_attr") else None
        edge_weight = None

        model.train()
        for epoch in epoch_iter:
            optimizer.zero_grad()

            loss = model.node_classification_loss(features, edge_index, edge_weight)
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss.item(): .4f}")

            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print("Early stopping!")
                break

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            embeds = model.embed(features, edge_index, edge_weight)

        nclass = int(torch.max(data.y) + 1)
        opt = {
            "idx_train": data.train_mask,
            "idx_val": data.val_mask,
            "idx_test": data.test_mask,
            "num_classes": nclass,
        }
        result = LogRegTrainer().train(embeds, data.y, opt)
        print(f"TestAcc: {result: .4f}")
        return dict(Acc=result)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class LogRegTrainer(object):
    def train(self, data, labels, opt):
        device = data.device
        idx_train = opt["idx_train"].to(device)
        idx_test = opt["idx_test"].to(device)
        nclass = opt["num_classes"]
        nhid = data.shape[-1]
        labels = labels.to(device)

        train_embs = data[idx_train]
        test_embs = data[idx_test]

        train_lbls = labels[idx_train]
        test_lbls = labels[idx_test]
        tot = 0

        xent = nn.CrossEntropyLoss()

        for _ in range(50):
            log = LogReg(nhid, nclass).to(device)
            optimizer = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.to(device)

            for _ in range(100):
                log.train()
                optimizer.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                optimizer.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            tot += acc.item()
        return tot / 50
