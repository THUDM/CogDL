import os
import os.path as osp
import time

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from cogdl.models.automl.gnn_net import GNNNet, MLP
from cogdl.models.automl.utils import TopAverage, EarlyStop
from cogdl.models.automl.search_space import SearchSpace


def load_data(dataset="Cora", supervised=False, full_data=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    assert dataset.lower() in ["cora", "citeseer", "pubmed"]
    dataset = Planetoid(path, dataset.capitalize(), T.NormalizeFeatures())
    return dataset


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class GNNManager(object):
    def __init__(self, args):
        self.args = args
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file
        self.shared_params = None

        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)

        self.loss_fn = torch.nn.functional.nll_loss

        self.data = load_data(args.dataset)
        self.args.num_features = self.data.num_features
        self.args.num_classes = self.data.num_classes
        self.data = self.data[0]
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.data.to(device)

        self.search_space = SearchSpace().get_search_space()
        self.search_keynums = len(self.search_space.keys())
        self.model_saved_path = args.model_saved_path
        if not osp.exists(self.model_saved_path):
            os.mkdir(self.model_saved_path)

    def build_srgnn(self, actions, in_features=None):
        # 1-layer GNN
        return GNNNet(actions, self.args, )

        # multi-layer GNN
        # assert in_features is not None
        # assert len(actions) % self.search_keynums == 0
        # num_features = self.args.num_features
        # num_classes = self.args.num_classes
        # nets = torch.nn.ModuleList()
        # for i in range(0, len(actions), self.search_keynums):
        #     self.args.num_features = in_features[i//self.search_keynums]
        #     nets.append(GNNNet(actions=actions[i:i + self.search_keynums], args=self.args))
        #     # print(self.args.num_features,
        #     #       actions[i + self.get_action_index("num_heads")] * actions[i + self.get_action_index("hidden_size")] *
        #     #       actions[i + self.get_action_index("num_hops")])
        #     # self.args.num_features = actions[i + self.get_action_index("num_heads")] \
        #     #                          * actions[i + self.get_action_index("hidden_size")] \
        #     #                                    * actions[i + self.get_action_index("num_hops")]
        #
        # nets.append(MLP(in_features[-1], num_classes, args=self.args))
        # self.args.num_features = num_features
        # return nets

    def get_action_index(self, action):
        index = list(self.search_space.keys()).index(action)
        assert index >= 0
        return index

    def update_args(self, args):
        self.args = args

    def train(self, actions=None):
        # if share_param:
        # 1-layer
        model = self.build_srgnn(actions, self.args)
        path = self.get_filename(actions, self.args.num_features)
        model = self.load_param(model, path)

        # multi-layer
        # in_features = [self.args.num_features]
        # for i in range(0, len(actions), self.search_keynums):
        #     # in_features.append(actions[i + self.get_action_index("num_heads")] \
        #     #                          * actions[i + self.get_action_index("hidden_size")] \
        #     #                                    * actions[i + self.get_action_index("num_hops")])
        #     in_features.append(actions[i + self.get_action_index("num_heads")]
        #                        * actions[i + self.get_action_index("hidden_size")])
        # model = self.build_srgnn(actions, in_features)
        #
        # paths = [self.get_filename(actions[i * self.search_keynums: (i + 1) * self.search_keynums], in_features[i])
        #          for i in range(len(model) - 1)]
        # paths.append(self.get_mlp_filename(actions[-self.search_keynums:], in_features[-1]))
        #
        # for i in range(len(model)):
        #     try:
        #         model[i] = self.load_param(model[i], paths[i])
        #     except RuntimeError as e:
        #         print(e)
        #         print("Error model: ", model[i])
        #         print(type(model[i]))
        #         print("path attempted : ", paths[i])
        #         print("All models: ", model)
        #         exit(0)
        # --------------------

        try:
            if self.args.cuda:
                model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda)
        except RuntimeError as e:
            if "cuda" in str(e).lower():
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)

        # 1 layer
        self.save_param(model, path, update_all=(reward > 0))

        # multi-layer
        # for i in range(len(model)):
        #     self.save_param(model[i], paths[i], update_all=(reward > 0))

        self.record_action_info(actions, reward, val_acc)

        return reward, val_acc

    def run_model(self, model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False):
        dur = []
        best_val_loss = 100000
        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(epochs):
            model.train()
            # 1 layer
            logits = model(data.x, data.edge_index, data.edge_attr)
            # ------

            # # multi-layer
            # hidden = model[0](data.x, data.edge_index, data.edge_attr)
            # for i in range(len(model) - 1):
            #     hidden = model[i + 1](hidden, data.edge_index, data.edge_attr)
            # logits = hidden
            # -----

            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            model.eval()

            # 1 layer
            logits = model(data.x, data.edge_index, data.edge_attr)
            # -----

            # # multi-layer
            # hidden = model[0](data.x, data.edge_index, data.edge_attr)
            # for i in range(len(model) - 1):
            #     hidden = model[i + 1](hidden, data.edge_index, data.edge_attr)
            # logits = hidden
            # -----

            train_acc = evaluate(logits, data.y, data.train_mask)
            val_acc = evaluate(logits, data.y, data.val_mask)
            val_loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            test_acc = evaluate(logits, data.y, data.test_mask)

            if val_loss < best_val_loss or val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc

            if self.early_stop_manager.should_save(train_loss, train_acc, val_loss, val_acc):
                break

            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))
        print(f"val_score:{best_val_acc},test_score:{best_test_acc}")
        if return_best:
            return model, best_val_acc, best_test_acc
        else:
            return model, best_val_acc

    def test_with_param(self, actions=None, with_retrain=False):
        return self.train(actions)

    def evaluate(self, actions):
        return self.train(actions)

    def record_action_info(self, origin_action, reward, val_acc):
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as file:
            # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")

    def save_param(self, model, path, update_all=False):
        # share param by layer
        if update_all:
            torch.save(model.state_dict(), path)
            return model
        # don't share parameters
        # pass

    def load_param(self, model, path, update_all=False):
        # share param by layer
        try:
            model.load_state_dict(torch.load(path))
            return model
        except FileNotFoundError:
            return model
        # don't share parameters
        # pass

    def get_filename(self, actions, in_feats):
        attn = actions[self.get_action_index("attention_type")]
        hidden_size = actions[self.get_action_index("hidden_size")]
        num_heads = actions[self.get_action_index("num_heads")]
        num_hops = actions[self.get_action_index("num_hops")]
        attn_att = actions[self.get_action_index("attention_type_att")]
        unique_model = f"gnn_att_{attn}_hidden_{hidden_size}_heads_{num_heads}_feats_{in_feats}_nhtop_{num_hops}_att_at_{attn_att}.pth"
        return osp.join(self.model_saved_path, unique_model)

    def get_mlp_filename(self, actions, in_featues):
        unique_model = f"mlp_{in_featues}.pth"
        return osp.join(self.model_saved_path, unique_model)
