from tqdm import tqdm
import copy
import torch

from cogdl.utils import symmetric_normalization, row_normalization, spmm
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

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.ppr_norm = args.ppr_norm if hasattr(args, "ppr_norm") else "sym"

    def preprocess_data(self, dataset):
        data = dataset[0]
        num_nodes = data.x.shape[0]
        nodes = torch.arange(num_nodes)
        self.train_idx = nodes[data.train_mask]
        self.test_idx = nodes[data.test_mask]
        self.val_idx = nodes[data.val_mask]
        train_topk_matrix = build_topk_ppr_matrix_from_data(
            data.edge_index, self.alpha, self.epsilon, self.train_idx.numpy(), self.topk, self.normalization
        )
        train_dataset = PPRGoDataset(data.x, train_topk_matrix, self.train_idx, data.y)

        val_topk_matrix = build_topk_ppr_matrix_from_data(
            data.edge_index, self.alpha, self.epsilon, self.val_idx.numpy(), self.topk, self.normalization
        )
        val_dataset = PPRGoDataset(data.x, val_topk_matrix, self.val_idx, data.y)
        return train_dataset, val_dataset

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
        train_dataset, val_dataset = self.preprocess_data(dataset)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = self.get_dataloader(train_dataset)
        val_loader = self.get_dataloader(val_dataset)

        best_loss = 1000
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
                f"Epoch: {epoch}, TrainLoss: {train_loss: .4f}, ValLoss: {best_loss: .4f}, ValAcc: {best_acc: .4f}"
            )
        self.model = best_model
        test_acc = self._test_step(dataset[0])
        return dict(Acc=test_acc)

    def _train_step(self, loader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        preds = []
        loss_items = []
        for batch in loader:
            x, targets, ppr_scores, y = [item.to(self.device) for item in batch]
            if is_train:
                loss = self.model.node_classification_loss(x, targets, ppr_scores, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    pred = self.model(x, targets, ppr_scores)
                    pred = torch.nn.functional.log_softmax(pred, dim=-1)
                    loss = torch.nn.functional.nll_loss(pred, y)

                    pred = pred.max(1)[1]
                    pred = pred.eq(y)
                    preds.append(pred)
            loss_items.append(loss.item())

        if is_train:
            return sum(loss_items) / len(loss_items)
        else:
            preds = torch.cat(preds)
            acc = preds.sum().item() / preds.shape[0]
            return acc, sum(loss_items) / len(loss_items)

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

        predictions = predictions.argmax(1)
        labels = data.y[data.test_mask]
        pred = predictions[data.test_mask]
        acc = pred.eq(labels).sum().item() / pred.shape[0]
        return acc
