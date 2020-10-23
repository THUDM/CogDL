import copy
import random
import os
import logging
import json
import numpy as np
import torch
from torch.optim import Adam, Adagrad, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss, BCELoss, KLDivLoss
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.datasets.triple_kg_data import KnowledgeGraph, BidirectionalOneShotIterator, TrainDataset



from . import BaseTask, register_task


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

@register_task("knowledge_link_prediction")
class KnowledgeLinkPrediction(BaseTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
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
    
    def __init__(self, args):
        super(KnowledgeLinkPrediction, self).__init__(args)
        self.dataset = build_dataset(args)
        args.nentity = self.dataset.num_entities
        args.nrelation = self.dataset.num_relations
        self.model = build_model(args)
        self.args = args
        set_logger(args)
        logging.info('Model: %s' % args.model)
        logging.info('#entity: %d' % args.nentity)
        logging.info('#relation: %d' % args.nrelation)

    def train(self):

        train_triples = self.dataset.tuple[self.dataset.train_start_idx:self.dataset.valid_start_idx]
        logging.info('#train: %d' % len(train_triples))
        valid_triples = self.dataset.tuple[self.dataset.valid_start_idx:self.dataset.test_start_idx]
        logging.info('#valid: %d' % len(valid_triples))
        test_triples = self.dataset.tuple[self.dataset.test_start_idx:]
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
        
            