from tqdm import tqdm
import copy
import os
import torch
import scipy.sparse as sp

from cogdl.utils import symmetric_normalization, row_normalization
from cogdl.layers.pprgo_modules import build_topk_ppr_matrix_from_data, PPRGoDataset


class PPRGoTrainer(object):
    def __init__(self, args):
        self.alpha = args.alpha
        self.topk = args.k
        self.epsilon = args.eps
        self.normalization = args.norm
        self.batch_size = args.batch_size
        if hasattr(args, "test_batch_size"):
            self.test_batch_size = args.test_batch_size
        else:
            self.test_batch_size = self.batch_size

        self.max_epoch = args.max_epoch
        self.patience = args.patience
        self.lr = args.lr
        self.eval_step = args.eval_step
        self.weight_decay = args.weight_decay
        self.dataset_name = args.dataset
        self.loss_func = None
        self.evaluator = None
        self.nprop_inference = args.nprop_inference

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.ppr_norm = args.ppr_norm if hasattr(args, "ppr_norm") else "sym"

    def ppr_run(self, dataset, mode="train"):
        data = dataset[0]
        num_nodes = data.x.shape[0]
        nodes = torch.arange(num_nodes)
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask
        index = nodes[mask].numpy()

        if mode == "train" and hasattr(data, "edge_index_train"):
            edge_index = data.edge_index_train
        else:
            edge_index = data.edge_index

        if not os.path.exists("./saved"):
            os.mkdir("saved")
        path = f"./saved/{self.dataset_name}_{self.topk}_{self.alpha}_{self.normalization}.{mode}.npz"

        if os.path.exists(path):
            print(f"Load {mode} from cached")
            train_topk_matrix = sp.load_npz(path)
        else:
            print(f"Fail to load {mode}")
            train_topk_matrix = build_topk_ppr_matrix_from_data(
                edge_index, self.alpha, self.epsilon, index, self.topk, self.normalization
            )
            sp.save_npz(path, train_topk_matrix)
        result = PPRGoDataset(data.x, train_topk_matrix, index, data.y)
        return result

    def get_dataloader(self, dataset):
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False,
            ),
            batch_size=None,
        )
        return data_loader

    def fit(self, model, dataset):
        self.loss_func, self.evaluator = dataset.get_evaluator()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_loader = self.get_dataloader(self.ppr_run(dataset, "train"))
        val_loader = self.get_dataloader(self.ppr_run(dataset, "val"))

        best_loss = 1000
        val_loss = 1000
        best_acc = 0
        best_model = None

        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            train_loss = self._train_step(train_loader, True)
            if (epoch + 1) % self.eval_step == 0:
                val_acc, val_loss = self._train_step(val_loader, False)
                if val_loss < best_loss:
                    best_acc = val_acc
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.model)
            epoch_iter.set_description(
                f"Epoch: {epoch}, TrainLoss: {train_loss: .4f}, ValLoss: {val_loss: .4f}, ValAcc: {best_acc: .4f}"
            )
            print()
        self.model = best_model

        del train_loader
        del val_loader
        if self.nprop_inference <= 0 or self.dataset_name == "ogbn-papers100M":
            test_loader = self.get_dataloader(self.ppr_run(dataset, "test"))
            test_acc, test_loss = self._train_step(test_loader, False)
        else:
            test_acc = self._test_step(dataset[0])

        print(f"TestAcc: {test_acc: .4f}")
        return dict(Acc=test_acc)

    def _train_step(self, loader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        preds = []
        loss_items = []
        labels = []
        for batch in loader:
            x, targets, ppr_scores, y = [item.to(self.device) for item in batch]
            if is_train:
                pred = self.model(x, targets, ppr_scores)
                if len(y.shape) == 1:
                    y = y.long()
                    pred = torch.nn.functional.log_softmax(pred, dim=-1)
                loss = self.loss_func(pred, y)
                if len(loss.shape) > 1:
                    loss = torch.sum(torch.mean(loss, dim=0))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    pred = self.model(x, targets, ppr_scores)
                    if len(y.shape) == 1:
                        y = y.long()
                        pred = torch.nn.functional.log_softmax(pred, dim=-1)
                    loss = self.loss_func(pred, y)
                    if len(loss.shape) > 1:
                        loss = torch.sum(torch.mean(loss, dim=0))

                    preds.append(pred)
                    labels.append(y)
            loss_items.append(loss.item())

        if is_train:
            return sum(loss_items) / len(loss_items)
        else:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            score = self.evaluator(labels, preds)
            return score, sum(loss_items) / len(loss_items)

    def _test_step(self, data):
        self.model.eval()

        if self.normalization == "sym":
            norm_func = symmetric_normalization
        elif self.normalization == "row":
            norm_func = row_normalization
        else:
            raise NotImplementedError

        with torch.no_grad():
            predictions = self.model.predict(data.x, data.edge_index, self.test_batch_size, norm_func)

        labels = data.y[data.test_mask]
        preds = predictions[data.test_mask]

        score = self.evaluator(labels, preds)
        return score
