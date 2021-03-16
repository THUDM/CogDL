from tqdm import tqdm

import torch
import torch.nn as nn
import torch.sparse

from .base_trainer import BaseTrainer

class GAETrainer(BaseTrainer):
    def __init__(self, args):
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.device = args.device_id[0] if not args.cpu else "cpu"

    @staticmethod
    def build_trainer_from_args(args):
        pass

    def fit(self, model, data):
        model = model.to(self.device)
        self.num_nodes = data.x.shape[0]

        print("Training initial embedding...")
        epoch_iter = tqdm(range(self.max_epoch))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in epoch_iter:
            model.train()
            optimizer.zero_grad()
            loss = model.make_loss(data)
            loss.backward()
            optimizer.step()
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

        return model
