import multiprocessing.pool as mp
import os
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from cogdl.data import Dataset
from cogdl.layers.gpt_gnn_module import (
    sample_subgraph,
    feature_reddit,
    to_torch,
    randint,
    GNN,
    load_gnn,
    Classifier,
    preprocess_dataset,
)
from cogdl.models.supervised_model import SupervisedHeterogeneousNodeClassificationModel
from cogdl.trainers.supervised_trainer import (
    SupervisedHomogeneousNodeClassificationTrainer,
    SupervisedHeterogeneousNodeClassificationTrainer,
)


graph_pool = None


def node_classification_sample(args, target_type, seed, nodes, time_range):
    """
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    """
    global graph_pool
    np.random.seed(seed)
    samp_nodes = np.random.choice(nodes, args.batch_size, replace=False)
    feature, times, edge_list, _, texts = sample_subgraph(
        graph_pool,
        time_range,
        inp={
            target_type: np.concatenate([samp_nodes, np.ones(args.batch_size)])
            .reshape(2, -1)
            .transpose()
        },
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        feature_extractor=feature_reddit,
    )

    (
        node_feature,
        node_type,
        edge_time,
        edge_index,
        edge_type,
        node_dict,
        edge_dict,
    ) = to_torch(feature, times, edge_list, graph_pool)

    x_ids = np.arange(args.batch_size)
    return (
        node_feature,
        node_type,
        edge_time,
        edge_index,
        edge_type,
        x_ids,
        graph_pool.y[samp_nodes],
    )


def prepare_data(
    args, graph, target_type, train_target_nodes, valid_target_nodes, pool
):
    """
        Sampled and prepare training and validation data using multi-process parallization.
    """
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(
            node_classification_sample,
            args=(args, target_type, randint(), train_target_nodes, {1: True}),
        )
        jobs.append(p)
    p = pool.apply_async(
        node_classification_sample,
        args=(args, target_type, randint(), valid_target_nodes, {1: True}),
    )
    jobs.append(p)
    return jobs


class GPT_GNNHomogeneousTrainer(SupervisedHomogeneousNodeClassificationTrainer):
    def __init__(self, args):
        super(GPT_GNNHomogeneousTrainer, self).__init__()
        self.args = args

    def fit(
        self, model: SupervisedHeterogeneousNodeClassificationModel, dataset: Dataset
    ) -> None:
        args = self.args
        self.device = args.device_id[0] if not args.cpu else "cpu"

        self.data = preprocess_dataset(dataset)

        global graph_pool
        graph_pool = self.data
        self.target_type = "def"
        self.train_target_nodes = self.data.train_target_nodes
        self.valid_target_nodes = self.data.valid_target_nodes
        self.test_target_nodes = self.data.test_target_nodes

        self.types = self.data.get_types()
        self.criterion = torch.nn.NLLLoss()

        self.stats = []
        self.res = []
        self.best_val = 0
        self.train_step = 0

        self.pool = mp.Pool(args.n_pool)
        self.st = time.time()
        self.jobs = prepare_data(
            args,
            self.data,
            self.target_type,
            self.train_target_nodes,
            self.valid_target_nodes,
            self.pool,
        )

        """
            Initialize GNN (model is specified by conv_name) and Classifier
        """
        self.gnn = GNN(
            conv_name=args.conv_name,
            in_dim=len(self.data.node_feature[self.target_type]["emb"].values[0]),
            n_hid=args.n_hid,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            num_types=len(self.types),
            num_relations=len(self.data.get_meta_graph()) + 1,
            prev_norm=args.prev_norm,
            last_norm=args.last_norm,
            use_RTE=False,
        )

        if args.use_pretrain:
            self.gnn.load_state_dict(
                load_gnn(torch.load(args.pretrain_model_dir)), strict=False
            )
            print("Load Pre-trained Model from (%s)" % args.pretrain_model_dir)

        self.classifier = Classifier(args.n_hid, self.data.y.max().item() + 1)

        self.model = torch.nn.Sequential(self.gnn, self.classifier).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)

        if args.scheduler == "cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                pct_start=0.02,
                anneal_strategy="linear",
                final_div_factor=100,
                max_lr=args.max_lr,
                total_steps=args.n_batch * args.n_epoch + 1,
            )
        elif args.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 500, eta_min=1e-6
            )
        else:
            assert False

        self.train_data = [job.get() for job in self.jobs[:-1]]
        self.valid_data = self.jobs[-1].get()
        self.pool.close()
        self.pool.join()

        self.et = time.time()
        print("Data Preparation: %.1fs" % (self.et - self.st))

        for epoch in np.arange(self.args.n_epoch) + 1:
            """
                Prepare Training and Validation Data
            """
            train_data = [job.get() for job in self.jobs[:-1]]
            valid_data = self.jobs[-1].get()
            self.pool.close()
            self.pool.join()
            """
                After the data is collected, close the pool and then reopen it.
            """
            self.pool = mp.Pool(self.args.n_pool)
            self.jobs = prepare_data(
                self.args,
                self.data,
                self.target_type,
                self.train_target_nodes,
                self.valid_target_nodes,
                self.pool,
            )
            self.et = time.time()
            print("Data Preparation: %.1fs" % (self.et - self.st))

            """
                Train
            """
            self.model.train()
            train_losses = []
            for (
                node_feature,
                node_type,
                edge_time,
                edge_index,
                edge_type,
                x_ids,
                ylabel,
            ) in train_data:
                node_rep = self.gnn.forward(
                    node_feature.to(self.device),
                    node_type.to(self.device),
                    edge_time.to(self.device),
                    edge_index.to(self.device),
                    edge_type.to(self.device),
                )
                res = self.classifier.forward(node_rep[x_ids])
                loss = self.criterion(res, ylabel.to(self.device))

                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                train_losses += [loss.cpu().detach().tolist()]
                self.train_step += 1
                self.scheduler.step(self.train_step)
                del res, loss
            """
                Valid
            """
            self.model.eval()
            with torch.no_grad():
                (
                    node_feature,
                    node_type,
                    edge_time,
                    edge_index,
                    edge_type,
                    x_ids,
                    ylabel,
                ) = valid_data
                node_rep = self.gnn.forward(
                    node_feature.to(self.device),
                    node_type.to(self.device),
                    edge_time.to(self.device),
                    edge_index.to(self.device),
                    edge_type.to(self.device),
                )
                res = self.classifier.forward(node_rep[x_ids])
                loss = self.criterion(res, ylabel.to(self.device))

                """
                    Calculate Valid F1. Update the best model based on highest F1 score.
                """
                valid_f1 = f1_score(
                    ylabel.tolist(), res.argmax(dim=1).cpu().tolist(), average="micro"
                )

                if valid_f1 > self.best_val:
                    self.best_val = valid_f1
                    # torch.save(
                    #     self.model,
                    #     os.path.join(
                    #         self.args.model_dir,
                    #         self.args.task_name + "_" + self.args.conv_name,
                    #     ),
                    # )
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    print("UPDATE!!!")

                self.st = time.time()
                print(
                    (
                        "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid F1: %.4f"
                    )
                    % (
                        epoch,
                        (self.st - self.et),
                        self.optimizer.param_groups[0]["lr"],
                        np.average(train_losses),
                        loss.cpu().detach().tolist(),
                        valid_f1,
                    )
                )
                self.stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
                del res, loss
            del train_data, valid_data

        self.model.load_state_dict(self.best_model_dict)
        best_model = self.model.to(self.device)
        # best_model = torch.load(
        #     os.path.join(
        #         self.args.model_dir, self.args.task_name + "_" + self.args.conv_name
        #     )
        # ).to(self.device)
        best_model.eval()
        gnn, classifier = best_model
        with torch.no_grad():
            test_res = []
            for _ in range(10):
                (
                    node_feature,
                    node_type,
                    edge_time,
                    edge_index,
                    edge_type,
                    x_ids,
                    ylabel,
                ) = node_classification_sample(
                    self.args,
                    self.target_type,
                    randint(),
                    self.test_target_nodes,
                    {1: True},
                )
                paper_rep = gnn.forward(
                    node_feature.to(self.device),
                    node_type.to(self.device),
                    edge_time.to(self.device),
                    edge_index.to(self.device),
                    edge_type.to(self.device),
                )[x_ids]
                res = classifier.forward(paper_rep)
                test_acc = accuracy_score(
                    ylabel.tolist(), res.argmax(dim=1).cpu().tolist()
                )
                test_res += [test_acc]
            return dict(Acc=np.average(test_res))
        #     # print("Best Test F1: %.4f" % np.average(test_res))

    @classmethod
    def build_trainer_from_args(cls, args):
        pass


class GPT_GNNHeterogeneousTrainer(SupervisedHeterogeneousNodeClassificationTrainer):
    def __init__(self, model, dataset):
        super(GPT_GNNHeterogeneousTrainer, self).__init__(model, dataset)

    def fit(self) -> None:
        raise NotImplemented

    def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
        raise NotImplemented
