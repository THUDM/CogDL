import os
import time

import torch
from . import BaseTask, register_task
from cogdl.models.automl.trainer import Trainer


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


@register_task("gnn_nas")
class GNNNAS(BaseTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training SRGCN_NAS, derive: Deriving Architectures')
        parser.add_argument('--save_epoch', type=int, default=2)
        parser.add_argument('--max_save_num', type=int, default=5)
        # controller
        parser.add_argument('--layers_of_child_model', type=int, default=2)
        parser.add_argument('--shared_initial_step', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
        parser.add_argument('--entropy_coeff', type=float, default=1e-4)
        parser.add_argument('--shared_rnn_max_length', type=int, default=35)
        parser.add_argument('--load_path', type=str, default='')
        parser.add_argument('--search_mode', type=str, default='macro')
        parser.add_argument('--format', type=str, default='two')
        parser.add_argument('--max_epoch', type=int, default=10)

        parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
        parser.add_argument('--discount', type=float, default=1.0)
        parser.add_argument('--controller_max_step', type=int, default=100,
                            help='step for controller parameters')
        parser.add_argument('--controller_optim', type=str, default='adam')
        parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                            help="will be ignored if --controller_lr_cosine=True")
        parser.add_argument('--controller_grad_clip', type=float, default=0)
        parser.add_argument('--tanh_c', type=float, default=2.5)
        parser.add_argument('--softmax_temperature', type=float, default=5.0)
        parser.add_argument('--derive_num_sample', type=int, default=100)
        parser.add_argument('--derive_finally', type=bool, default=True)
        parser.add_argument('--model_saved_path', type=str, default=".")


        # child model
        #parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
        #                    help="The input dataset.")
        parser.add_argument("--epochs", type=int, default=300,
                            help="number of training epochs")
        parser.add_argument("--retrain_epochs", type=int, default=300,
                            help="number of training epochs")
        parser.add_argument("--multi_label", type=bool, default=False,
                            help="multi_label or single_label task")
        parser.add_argument("--residual", action="store_false",
                            help="use residual connection")
        parser.add_argument("--in-drop", type=float, default=0.6,
                            help="input feature dropout")
        parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
        parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                            help="learning rate")
        parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                            help="optimizer save path")
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--max_param', type=float, default=5E6)
        parser.add_argument('--supervised', type=bool, default=False)
        parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")
    
    def __init__(self, args):
        super(GNNNAS, self).__init__(args)
        args.cuda = torch.cuda.is_available() and not args.cpu
        # args.max_epoch = 1
        # args.controller_max_step = 1
        # args.derive_num_sample = 1
        self.args = args

        makedirs(self.args.dataset)
        self.trainer = Trainer(self.args)

    def train(self):
        if self.args.mode == 'train':
            self.trainer.train()
        elif self.args.mode == 'derive':
            self.trainer.derive()
        else:
            raise Exception(f"[!] Mode not found: {self.args.mode}")
    
