import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from .base_trainer import BaseTrainer


class SelfSupervisedTrainer(BaseTrainer):
    def __init__(self, args):
        super(SelfSupervisedTrainer, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.epochs = args.max_epoch
        self.patience = args.patience
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.save_dir = args.save_dir
        self.load_emb_path = args.load_emb_path

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def fit(self, model, data):
        self.data = data
        self.data.edge_attr = torch.ones(data.edge_index.shape[1]).to(self.device)
        self.data.apply(lambda x: x.to(self.device))

        if self.load_emb_path is not None:
            embeds = np.load(self.load_emb_path)
            embeds = torch.from_numpy(embeds).to(self.device)
            return self.evaluate(embeds)

        best = 1e9
        cnt_wait = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

        epoch_iter = tqdm(range(self.epochs))
        model = model.to(self.device)

        model.train()
        for epoch in epoch_iter:
            optimizer.zero_grad()

            loss = model.node_classification_loss(data)
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
            embeds = model.embed(data)
        self.save_embed(embeds)

        return self.evaluate(embeds)

    def evaluate(self, embeds):
        nclass = int(torch.max(self.data.y) + 1)
        opt = {
            "idx_train": self.data.train_mask,
            "idx_val": self.data.val_mask,
            "idx_test": self.data.test_mask,
            "num_classes": nclass,
        }
        result = LogRegTrainer().train(embeds, self.data.y, opt)
        print(f"TestAcc: {result: .4f}")
        return dict(Acc=result)

    def save_embed(self, embed):
        os.makedirs(self.save_dir, exist_ok=True)
        embed = embed.cpu().numpy()
        out_file = os.path.join(self.save_dir, f"{self.model_name}_{self.dataset_name}.npy")
        np.save(out_file, embed)


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
