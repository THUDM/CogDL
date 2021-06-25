import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn

from cogdl.data.sampler import get_sampler
from cogdl.trainers.sampled_trainer import SAINTTrainer
from .base_trainer import BaseTrainer
from . import register_trainer, TRAINER_REGISTRY


def batch_iterator(sampler, dataset, ops):
    loader = get_sampler(sampler, dataset, ops)
    while True:
        if sampler in ["node", "edge", "rw", "mrw"]:
            yield sampler.one_batch()
        else:
            loader.shuffle()
            for batch in loader:
                yield batch


def get_args4sampler(args):
    trainer = TRAINER_REGISTRY[args.sampler]
    settings = dict(batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    args4sampler = trainer.get_args4sampler(args)
    args4sampler.update(settings)
    return args4sampler


@register_trainer("self_supervised")
class SelfSupervisedTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--sampling", action="store_true")
        args = parser.parse_args()
        if args.sampling:
            parser.add_argument("--sampler", type=str, default="node")
            args = parser.parse_args()
            if args.sampler in ["node", "edge", "rw", "mrw"]:
                SAINTTrainer.add_args(parser)
            else:
                TRAINER_REGISTRY[args.sampler].add_args(parser)
        # fmt: on

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(SelfSupervisedTrainer, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.epochs = args.max_epoch
        self.patience = args.patience
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.save_dir = args.save_dir
        self.load_emb_path = args.load_emb_path
        self.lr = args.lr
        if args.sampling:
            self.args4sampler = get_args4sampler(args)
            self.sampler = args.sampler
        else:
            self.args4sampler = None

    def fit(self, model, dataset, evaluate=True):
        data = dataset.data
        data.add_remaining_self_loops()
        self.data = data
        if self.load_emb_path is not None:
            embeds = np.load(self.load_emb_path)
            embeds = torch.from_numpy(embeds).to(self.device)
            return self.evaluate(embeds)

        if self.args4sampler:
            batch_iter = get_sampler(self.sampler, dataset, self.args4sampler)
        else:
            batch_iter = None
            data = data.to(self.device)
        best = 1e9
        cnt_wait = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)

        epoch_iter = tqdm(range(self.epochs))
        model = model.to(self.device)

        model.train()
        for epoch in epoch_iter:
            if batch_iter is None:
                batch = data
            else:
                batch = next(batch_iter).to(self.device)

            optimizer.zero_grad()

            loss = model.node_classification_loss(batch)
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

        self.data = data.to(self.device)
        with torch.no_grad():
            embeds = model.embed(self.data)
        self.save_embed(embeds)

        if not evaluate:
            return embeds

        return self.evaluate(embeds, dataset.get_loss_fn(), dataset.get_evaluator())

    def evaluate(self, embeds, loss_fn=None, evaluator=None):
        nclass = int(torch.max(self.data.y) + 1)
        opt = {
            "idx_train": self.data.train_mask.to(self.device),
            "idx_val": self.data.val_mask.to(self.device),
            "idx_test": self.data.test_mask.to(self.device),
            "num_classes": nclass,
        }
        result = LogRegTrainer().train(embeds, self.data.y.to(self.device), opt, loss_fn, evaluator)
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
    def train(self, data, labels, opt, loss_fn=None, evaluator=None):
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

        xent = nn.CrossEntropyLoss() if loss_fn is None else loss_fn

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
            if evaluator is None:
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            else:
                acc = evaluator(logits, test_lbls)
            tot += acc
        return tot / 50
