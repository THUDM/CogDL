import random

import os
import logging
import json
import copy
import networkx as nx
import numpy as np
import torch
from torch import mode
from torch.optim import Adam, Adagrad, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss, BCELoss, KLDivLoss
from torch.utils.data import WeightedRandomSampler
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task

from cogdl.datasets.kg_data import KnowledgeGraphDataset, BidirectionalOneShotIterator, TrainDataset

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def divide_data(input_list, division_rate):
    local_division = len(input_list) * np.cumsum(np.array(division_rate))
    random.shuffle(input_list)
    return [
        input_list[
            int(round(local_division[i - 1]))
            if i > 0
            else 0 : int(round(local_division[i]))
        ]
        for i in range(len(local_division))
    ]


def randomly_choose_false_edges(nodes, true_edges, num):
    true_edges_set = set(true_edges)
    tmp_list = list()
    all_flag = False
    for _ in range(num):
        trial = 0
        while True:
            x = nodes[random.randint(0, len(nodes) - 1)]
            y = nodes[random.randint(0, len(nodes) - 1)]
            trial += 1
            if trial >= 1000:
                all_flag = True
                break
            if x != y and (x, y) not in true_edges_set and (y, x) not in true_edges_set:
                tmp_list.append((x, y))
                break
        if all_flag:
            break
    return tmp_list


def gen_node_pairs(train_data, test_data, negative_ratio=5):
    G = nx.Graph()
    G.add_edges_from(train_data)

    training_nodes = set(list(G.nodes()))
    test_true_data = []
    for u, v in test_data:
        if u in training_nodes and v in training_nodes:
            test_true_data.append((u, v))
    test_false_data = randomly_choose_false_edges(
        list(training_nodes), train_data, len(test_data) * negative_ratio
    )
    return (test_true_data, test_false_data)


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def evaluate(embs, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    for edge in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def select_task(model_name=None, model=None):
    assert model_name is not None or model is not None
    if model_name is not None:
        if model_name in ["rgcn", "compgcn"]:
            return "KGLinkPrediction"
        if model_name in ["distmult", "transe", "rotate", "complex"]:
            return "TripleLinkPrediction"
        else:
            return "HomoLinkPrediction"
    else:
        from cogdl.models.nn import rgcn, compgcn
        from cogdl.models.emb import distmult, rotate, transe, complex
        if type(model) in [rgcn.LinkPredictRGCN, compgcn.LinkPredictCompGCN]:
            return "KGLinkPrediction"
        if type(model) in [distmult.DistMult, rotate.RotatE, transe.TransE, complex.ComplEx]:
            return "TripleLinkPrediction"
        else:
            return "HomoLinkPrediction"

class HomoLinkPrediction(nn.Module):
    def __init__(self, args, dataset=None, model=None):
        super(HomoLinkPrediction, self).__init__()
        dataset = build_dataset(args) if dataset is None else dataset
        data = dataset[0]
        self.data = data
        if hasattr(dataset, "num_features"):
            args.num_features = dataset.num_features
        model = build_model(args) if model is None else model
        self.model = model
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        edge_list = self.data.edge_index.numpy()
        edge_list = list(zip(edge_list[0], edge_list[1]))
        edge_set = set()
        for edge in edge_list:
            if (edge[0], edge[1]) not in edge_set and (edge[1], edge[0]) not in edge_set:
                edge_set.add(edge)
        edge_list = list(edge_set)
        self.train_data, self.test_data = divide_data(
            edge_list, [0.90, 0.10]
        )

        self.test_data = gen_node_pairs(
            self.train_data, self.test_data, args.negative_ratio
        )

    def train(self):
        G = nx.Graph()
        G.add_edges_from(self.train_data)
        embeddings = self.model.train(G)

        embs = dict()
        for vid, node in enumerate(G.nodes()):
            embs[node] = embeddings[vid]

        roc_auc, f1_score, pr_auc = evaluate(embs, self.test_data[0], self.test_data[1])
        print(
            f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}"
        )
        return dict(ROC_AUC=roc_auc, PR_AUC=pr_auc, F1=f1_score)

class TripleLinkPrediction(nn.Module):
    """
    Training process borrowed from `KnowledgeGraphEmbedding<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>`
    """
    def __init__(self, args, dataset=None, model=None):
        super(TripleLinkPrediction, self).__init__()
        self.dataset = build_dataset(args) if dataset is None else dataset
        args.nentity = self.dataset.num_entities
        args.nrelation = self.dataset.num_relations
        self.model = build_model(args) if model is None else model
        self.args = args
        set_logger(args)
        logging.info('Model: %s' % args.model)
        logging.info('#entity: %d' % args.nentity)
        logging.info('#relation: %d' % args.nrelation)

    def train(self):

        train_triples = self.dataset.triples[self.dataset.train_start_idx:self.dataset.valid_start_idx]
        logging.info('#train: %d' % len(train_triples))
        valid_triples = self.dataset.triples[self.dataset.valid_start_idx:self.dataset.test_start_idx]
        logging.info('#valid: %d' % len(valid_triples))
        test_triples = self.dataset.triples[self.dataset.test_start_idx:]
        logging.info('#test: %d' % len(test_triples))

        all_true_triples = train_triples + valid_triples + test_triples
        nentity, nrelation = self.args.nentity, self.args.nrelation

        if torch.cuda.is_available():
            self.args.cuda = True
            self.model = self.model.cuda()

        if self.args.do_train:
        # Set training dataloader iterator
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, self.args.negative_sample_size, 'head-batch'), 
                batch_size=self.args.batch_size,
                shuffle=True, 
                collate_fn=TrainDataset.collate_fn
            )
        
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, self.args.negative_sample_size, 'tail-batch'), 
                batch_size=self.args.batch_size,
                shuffle=True, 
                collate_fn=TrainDataset.collate_fn
            )
        
            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
            current_learning_rate = self.args.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=current_learning_rate
            )
            if self.args.warm_up_steps:
                warm_up_steps = self.args.warm_up_steps
            else:
                warm_up_steps = self.args.max_steps // 2

        if self.args.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % self.args.init_checkpoint)
            checkpoint = torch.load(os.path.join(self.args.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.args.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info('Ramdomly Initializing %s Model...' % self.args.model)
            init_step = 0
    
        step = init_step
    
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('batch_size = %d' % self.args.batch_size)
        logging.info('negative_adversarial_sampling = %d' % self.args.negative_adversarial_sampling)
        logging.info('hidden_dim = %d' % self.args.embedding_size)
        logging.info('gamma = %f' % self.args.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(self.args.negative_adversarial_sampling))
        if self.args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % self.args.adversarial_temperature)
    
        # Set valid dataloader as it would be evaluated during training
    
        if self.args.do_train:
            logging.info('learning_rate = %d' % current_learning_rate)

            training_logs = []
        
            #Training Loop
            for step in range(init_step, self.args.max_steps):
            
                log = self.model.train_step(self.model, optimizer, train_iterator, self.args)
            
                training_logs.append(log)
            
                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3
            
                if step % self.args.save_checkpoint_steps == 0:
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(self.model, optimizer, save_variable_list, self.args)
                
                if step % self.args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                    log_metrics('Training average', step, metrics)
                    training_logs = []
                
                if self.args.do_valid and step % self.args.valid_steps == 0:
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = self.model.test_step(self.model, valid_triples, all_true_triples, self.args)
                    log_metrics('Valid', step, metrics)
        
            save_variable_list = {
                'step': step, 
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(self.model, optimizer, save_variable_list, self.args)
        
        if self.args.do_valid:
            logging.info('Evaluating on Valid Dataset...')
            metrics = self.model.test_step(self.model, valid_triples, all_true_triples, self.args)
            log_metrics('Valid', step, metrics)

        logging.info('Evaluating on Test Dataset...')
        return self.model.test_step(self.model, test_triples, all_true_triples, self.args)

class KGLinkPrediction(nn.Module):
    def __init__(self, args, dataset=None, model=None):
        super(KGLinkPrediction, self).__init__()
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.evaluate_interval = args.evaluate_interval
        dataset = build_dataset(args) if dataset is None else dataset
        self.data = dataset[0]
        self.data.apply(lambda x: x.to(self.device))
        args.num_entities = len(torch.unique(self.data.edge_index))
        args.num_rels = len(torch.unique(self.data.edge_attr))
        model = build_model(args) if model is None else model
        self.model = model.to(self.device)
        self.max_epoch = args.max_epoch
        self.patience = min(args.patience, 20)
        self.grad_norm = 1.0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_mrr = 0
        best_model = None
        val_mrr = 0

        for epoch in epoch_iter:
            loss_n = self._train_step()
            if (epoch + 1) % self.evaluate_interval == 0:
                torch.cuda.empty_cache()
                val_mrr, _ = self._test_step("val")
                if val_mrr > best_mrr:
                    best_mrr = val_mrr
                    best_model = copy.deepcopy(self.model)
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        self.model = best_model
                        epoch_iter.close()
                        break
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, TrainLoss: {loss_n: .4f}, Val MRR: {val_mrr: .4f}, Best MRR: {best_mrr: .4f}"
            )
        self.model = best_model
        test_mrr, test_hits = self._test_step("test")
        print(
            f"Test MRR:{test_mrr}, Hits@1/3/10: {test_hits}"
        )
        return dict(MRR=test_mrr, HITS1=test_hits[0], HITS3=test_hits[1], HITS10=test_hits[2])

    def _train_step(self, split="train"):
        self.model.train()
        self.optimizer.zero_grad()
        loss_n = self.model.loss(self.data)
        loss_n.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        return loss_n.item()
    
    def _test_step(self, split="val"):
        self.model.eval()
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        edge_index = self.data.edge_index[:, mask]
        edge_attr = self.data.edge_attr[mask]
        mrr, hits = self.model.predict(edge_index, edge_attr)
        return mrr, hits

@register_task("link_prediction")
class LinkPrediction(BaseTask):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--evaluate-interval", type=int, default=30)
        parser.add_argument("--max-epoch", type=int, default=3000)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight-decay", type=float, default=0)
        
        parser.add_argument("--hidden-size", type=int, default=200) # KG
        parser.add_argument("--negative-ratio", type=int, default=5)
        
        # some arguments for triple-based knowledge graph embedding
        parser.add_argument('--cuda', action='store_true', help='use GPU')
        parser.add_argument('--do_train', action='store_true')
        parser.add_argument('--do_valid', action='store_true')
        parser.add_argument('-de', '--double_entity_embedding', action='store_true')
        parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
        
        parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
        parser.add_argument('-d', '--embedding_size', default=500, type=int)
        parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
        parser.add_argument('-g', '--gamma', default=12.0, type=float)
        parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
        parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
        parser.add_argument('-b', '--batch_size', default=1024, type=int)
        parser.add_argument('-r', '--regularization', default=0.0, type=float)
        parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
        parser.add_argument('--uni_weight', action='store_true', 
                            help='Otherwise use subsampling weighting like in word2vec')
        
        parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
        parser.add_argument('-save', '--save_path', default=None, type=str)
        parser.add_argument('--max_steps', default=100000, type=int)
        parser.add_argument('--warm_up_steps', default=None, type=int)
        
        parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
        parser.add_argument('--valid_steps', default=10000, type=int)
        parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
        parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(LinkPrediction, self).__init__(args)

        task_type = select_task(args.model, model)
        if task_type == "HomoLinkPrediction":
            self.task = HomoLinkPrediction(args, dataset, model)
        elif task_type == "KGLinkPrediction":
            self.task = KGLinkPrediction(args, dataset, model)
        elif task_type == "TripleLinkPrediction":
            self.task = TripleLinkPrediction(args, dataset, model)
    
    def train(self):
        return self.task.train()
