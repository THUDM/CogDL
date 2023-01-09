'''
This code is borrowed from https://github.com/Namkyeong/BGRL_Pytorch
'''

import numpy as np

import torch


import models
import utils
import data
import os
import sys
import warnings

from torch import optim
from tensorboardX import SummaryWriter

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize

warnings.filterwarnings("ignore")
torch.manual_seed(0)


class ModelTrainer:

    def __init__(self, args):
        self._args = args

        self._init()
        self.writer = SummaryWriter(log_dir="saved/BGRL_dataset({})".format(args.name))

    def _init(self):
        args = self._args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        # self._dataset = data.Dataset(root=args.root, name=args.name)[0]
        self._dataset = data.get_data(args.name)
        print(f"Data: {self._dataset}")
        hidden_layers = [int(dim) for dim in args.layers]
        layers = [self._dataset.x.shape[1]] + hidden_layers
        self._model = models.BGRL(layer_config=layers, pred_hid=args.pred_hid, dropout=args.dropout, epochs=args.epochs).to(self._device)
        print(self._model)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)

        # learning rate
        def lr_scheduler(epoch):
            if epoch <= args.warmup_epochs:
                return epoch / args.warmup_epochs
            else:
                return (1 + np.cos((epoch - args.warmup_epochs) * np.pi / (self._args.epochs - args.warmup_epochs))) * 0.5
        # lr_scheduler = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs \
        #             else ( 1 + np.cos((epoch - args.warmup_epochs) * np.pi / (self._args.epochs - args.warmup_epochs))) * 0.5

        self._scheduler = optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lr_scheduler)

    def train(self):
        # get initial test results
        print(self._args)
        print("start training!")

        print("Initial Evaluation...")
        self.infer_embeddings()
        test_best, test_std_best = self.evaluate()
        print("test: {:.4f}".format(test_best))

        # start training
        self._model.train()
        for epoch in range(self._args.epochs):
            
            self._dataset.to(self._device)

            augmentation = utils.Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]), float(self._args.aug_params[2]), float(self._args.aug_params[3]))
            view1, view2 = augmentation._feature_masking(self._dataset, self._device)

            v1_output, v2_output, loss = self._model(
                x1=view1.x, x2=view2.x, graph_v1=view1, graph_v2=view2,
                edge_weight_v1=view1.edge_attr, edge_weight_v2=view2.edge_attr)
                
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._model.update_moving_average()
            sys.stdout.write('\rEpoch {}/{}, loss {:.4f}, lr {}'.format(epoch + 1, self._args.epochs, loss.data, self._optimizer.param_groups[0]['lr']))
            sys.stdout.flush()
            
            if (epoch + 1) % self._args.cache_step == 0:
                print("")
                print("\nEvaluating {}th epoch..".format(epoch + 1))
                
                self.infer_embeddings()
                test_acc, test_std = self.evaluate()

                self.writer.add_scalar("stats/learning_rate", self._optimizer.param_groups[0]["lr"] , epoch + 1)
                self.writer.add_scalar("accs/test_acc", test_acc, epoch + 1)
                print("test: {:.4f} \n".format(test_acc))

        print()
        print("Training Done!")

    def infer_embeddings(self):
        
        self._model.train(False)
        self._embeddings = self._labels = None

        self._dataset.to(self._device)
        v1_output, v2_output, _ = self._model(
            x1=self._dataset.x, x2=self._dataset.x,
            graph_v1=self._dataset,
            graph_v2=self._dataset,
            edge_weight_v1=self._dataset.edge_attr,
            edge_weight_v2=self._dataset.edge_attr)
        emb = v1_output.detach()
        y = self._dataset.y.detach()
        if self._embeddings is None:
            self._embeddings, self._labels = emb, y
        else:
            self._embeddings = torch.cat([self._embeddings, emb])
            self._labels = torch.cat([self._labels, y])
                
    def evaluate(self):
        """
        Used for producing the results of Experiment 3.2 in the BGRL paper. 
        """
        test_accs = []
        
        self._embeddings = self._embeddings.cpu().numpy()
        self._labels = self._labels.cpu().numpy()
        self._dataset.to(torch.device("cpu"))

        one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
        self._labels = one_hot_encoder.fit_transform(self._labels.reshape(-1, 1)).astype(np.bool)

        self._embeddings = normalize(self._embeddings, norm='l2')
        
        for i in range(20):

            self._train_mask = self._dataset.train_mask[i]
            self._dev_mask = self._dataset.val_mask[i]
            if self._args.name in ["WikiCS"]:
                self._test_mask = self._dataset.test_mask
            else:
                self._test_mask = self._dataset.test_mask[i]

            # grid search with one-vs-rest classifiers
            best_test_acc, best_acc = 0, 0
            
            for c in 2.0 ** np.arange(-10, 11):
                clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
                clf.fit(self._embeddings[self._train_mask], self._labels[self._train_mask])

                y_pred = clf.predict_proba(self._embeddings[self._dev_mask])
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                val_acc = metrics.accuracy_score(self._labels[self._dev_mask], y_pred)
                if val_acc > best_acc:
                    best_acc = val_acc
                    y_pred = clf.predict_proba(self._embeddings[self._test_mask])
                    y_pred = np.argmax(y_pred, axis=1)
                    y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                    best_test_acc = metrics.accuracy_score(self._labels[self._test_mask], y_pred)
            test_accs.append(best_test_acc)
        return np.mean(test_accs), np.std(test_accs)


def train_eval(args):
    trainer = ModelTrainer(args)
    trainer.train()    
    trainer.writer.close()


def main():
    args = utils.parse_args()
    train_eval(args)


if __name__ == "__main__":
    main()