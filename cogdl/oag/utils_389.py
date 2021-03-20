import sys
import argparse
import threading
from tqdm import tqdm
import os
from datetime import datetime
import torch


def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument(
        "--config-file",
        "--cf",
        help="pointer to the configuration file of the experiment",
        type=str,
        required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )

    parser.add_argument(
        "--load_model_weights",
        default=None,
        type=str,
        required=False,
        help="The model weights to load"
    )

    # Optional Params
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=80,
        type=int,
        help=
        "The maximum number of masked tokens in a sequence to be predicted.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument(
        "--do_lower_case",
        default=True,
        action='store_true',
        help=
        "Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--use_pretrain',
                        default=False,
                        action='store_true',
                        help="Whether to use Bert Pretrain Weights or not")

    parser.add_argument(
        '--refresh_bucket_size',
        type=int,
        default=1,
        help=
        "This param makes sure that a certain task is repeated for this time steps to \
                            optimise on the back propogation speed with APEX's DistributedDataParallel"
    )
    parser.add_argument('--finetune',
                        default=False,
                        action='store_true',
                        help="Whether to finetune only")

    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='LE',
        help=
        'Choices LE, EE, EP (L: Linear, E: Exponetial, P: Polynomial warmup and decay)'
    )

    parser.add_argument('--lr_offset',
                        type=float,
                        default=0.0,
                        help='Offset added to lr.')

    parser.add_argument(
        '--load_training_checkpoint',
        '--load_cp',
        type=str,
        default=None,
        help=
        "This is the path to the TAR file which contains model+opt state_dict() checkpointed."
    )
    parser.add_argument(
        '--load_checkpoint_id',
        '--load_cp_id',
        type=str,
        default=None,
        help='Checkpoint identifier to load from checkpoint path')
    parser.add_argument(
        '--job_name',
        type=str,
        required=True,
        help="This is the path to store the output and TensorBoard results.")

    parser.add_argument(
        '--rewarmup',
        default=False,
        action='store_true',
        help='Rewarmup learning rate after resuming from a checkpoint')

    parser.add_argument(
        '--max_steps',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size to complete.'
    )

    parser.add_argument(
        '--max_steps_per_epoch',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size within an epoch to complete.'
    )

    parser.add_argument('--print_steps',
                        type=int,
                        default=100,
                        help='Interval to print training details.')

    parser.add_argument(
        '--data_path_prefix',
        type=str,
        default="",
        help=
        "Path to prefix data loading, helpful for AML and other environments")

    parser.add_argument(
        '--validation_data_path_prefix',
        type=str,
        default=None,
        help=
        "Path to prefix validation data loading, helpful if pretraining dataset path is different"
    )

    parser.add_argument('--deepspeed_transformer_kernel',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')

    parser.add_argument(
        '--stochastic_mode',
        default=False,
        action='store_true',
        help='Use stochastic mode for high-performance transformer kernel.')

    parser.add_argument(
        '--ckpt_to_save',
        nargs='+',
        type=int,
        help=
        'Indicates which checkpoints to save, e.g. --ckpt_to_save 160 161, by default all checkpoints are saved.'
    )

    parser.add_argument(
        '--attention_dropout_checkpoint',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.'
    )
    parser.add_argument(
        '--normalize_invertible',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.'
    )
    parser.add_argument(
        '--gelu_checkpoint',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.'
    )
    parser.add_argument('--deepspeed_sparse_attention',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed sparse self attention.')

    parser.add_argument('--use_masked_lm',
                        default=False,
                        action='store_true',
                        help='Use BertForPreTrainingPreLN (Default False) or BertForMaskedLM')

    return parser


def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
            (global_steps >= args.max_steps)

import os
from termcolor import colored
import h5py
from transformers import BertTokenizerFast
import re
tokenizer = BertTokenizerFast.from_pretrained('/mnt/V2OAG/models/oagbert_v1/model')

def print_instance(batch, predictions=None, _tokenizer=None, prediction_scores=None):
    if _tokenizer is None:
        _tokenizer = tokenizer
    token_type_str_lookup = ['TEXT', 'AUTHOR', 'VENUE', 'AFF', 'FOS']
    COLORS = ['white', 'green', 'blue', 'red', 'yellow']
    try:
        (termwidth, termheight) = os.get_terminal_size()
    except:
        termwidth, termheight = 200, 100
    input_ids = batch[1]
    token_type_ids = batch[3]
    input_mask = batch[2]
    masked_lm_labels = batch[4]
    position_ids = batch[5]
    position_ids_second = batch[6]
    inputs = {
        'input_ids': batch[1][0].cpu().detach().numpy(),
        'input_mask': batch[2][0].cpu().detach().numpy(),
        'token_type_ids': batch[3][0].cpu().detach().numpy(),
        'masked_lm_labels': batch[4][0].cpu().detach().numpy(),
        'position_ids': batch[5][0].cpu().detach().numpy(),
        'position_ids_second': batch[6][0].cpu().detach().numpy(),
        'predictions': predictions.cpu().detach().numpy() if predictions is not None else None
    }
    K = inputs['predictions'].shape[1] if predictions is not None else 0
    input_ids = [token_id for i, token_id in enumerate(inputs['input_ids']) if inputs['input_mask'][i].sum() > 0]
    position_ids = [position_id for i, position_id in enumerate(inputs['position_ids']) if inputs['input_mask'][i].sum() > 0]
    position_ids_second = [position_id for i, position_id in enumerate(inputs['position_ids_second']) if inputs['input_mask'][i].sum() > 0]
    token_type_ids = [token_type_id for i, token_type_id in enumerate(inputs['token_type_ids']) if inputs['input_mask'][i].sum() > 0]
    masks = [0 for i in inputs['input_ids']]
    prediction_topks = [[0 for i in inputs['input_ids']] for _ in range(K)]
    mask_indices = []
    for lm_pos, lm_id in enumerate(inputs['masked_lm_labels']):
        if lm_id < 0:
            continue
        masks[lm_pos] = lm_id
        mask_indices.append(lm_pos)
    for k in range(K):
        for lm_pos, token_id in zip(mask_indices, inputs['predictions'][:,k]):
            prediction_topks[k][lm_pos] = token_id
    input_tokens = _tokenizer.convert_ids_to_tokens(input_ids)
    masks_tokens = _tokenizer.convert_ids_to_tokens(masks)
    prediction_tokens = [_tokenizer.convert_ids_to_tokens(prediction_topks[k]) for k in range(K)]
    if prediction_scores is not None:
        for k in range(K):
            _idx = 0
            for _i, tok in enumerate(prediction_tokens[k]):
                if tok != '[PAD]':
                    prediction_tokens[k][_i] += '(%.4f)' % torch.exp(prediction_scores[_idx][k])
                    _idx += 1
    input_tokens_str = ['']
    position_ids_str = ['']
    position_ids_second_str = ['']
    token_type_ids_str = ['']
    masks_str = ['']
    prediction_topk_strs = [[''] for _ in range(K)]
    current_length = 0
    for pos, (input_token, position_id, position_id_second, token_type_id, mask) in enumerate(zip(input_tokens, position_ids, position_ids_second, token_type_ids, masks_tokens)):
        token_type = token_type_str_lookup[token_type_id]
        length = max(len(input_token) + 1, 7, len(token_type) + 1, len(mask) + 1, *[len(prediction_tokens[k][pos]) + 1 for k in range(K)])
        if current_length + length > termwidth:
            current_length = 0
            input_tokens_str.append('')
            position_ids_str.append('')
            position_ids_second_str.append('')
            token_type_ids_str.append('')
            masks_str.append('')
            for k in range(K):
                prediction_topk_strs[k].append('')
        current_length += length
        input_tokens_str[-1] += colored(input_token.rjust(length), COLORS[token_type_id])
        position_ids_str[-1] += str(position_id).rjust(length)
        position_ids_second_str[-1] += str(position_id_second).rjust(length)
        token_type_ids_str[-1] += token_type.rjust(length)
        masks_str[-1] += colored(mask.rjust(length) if mask != '[PAD]' else ''.rjust(length), COLORS[token_type_id])
        for k in range(K):
            v = prediction_tokens[k][pos] if prediction_tokens[k][pos] != '[PAD]' else ''
            prediction_topk_strs[k][-1] += colored(v.rjust(length), 'magenta' if v != mask and mask != '[CLS]' else 'cyan')

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    size = len(masks_str)
    for i in range(size):
        print()
        print(input_tokens_str[i])
        print(position_ids_str[i])
        print(position_ids_second_str[i])
        print(token_type_ids_str[i])
        if ansi_escape.sub('', masks_str[i]).strip() != '':
            print(masks_str[i])
        for k in range(K):
            if ansi_escape.sub('', prediction_topk_strs[k][i]).strip() != '':
                print(prediction_topk_strs[k][i])
        print('-' * termwidth)












class MultiProcessTqdm(object):

    def __init__(self, lock, positions, max_pos=100, update_interval=1000000, leave=False, fixed_pos=False, pos=None):
        self.lock = lock
        self.positions = positions
        self.max_pos = max_pos
        self.update_interval = update_interval
        self.leave = leave
        self.pbar = None
        self.pos = pos
        self.fixed_pos = fixed_pos

    def open(self, name, **kwargs):
        with self.lock:
            if self.pos is None or not self.fixed_pos:
                self.pos = 0
                while self.pos in self.positions:
                    self.pos += 1
                self.positions[self.pos] = name
            self.pbar = tqdm(position=self.pos % self.max_pos, leave=self.leave, desc='[%2d] %s' % (self.pos, name), **kwargs)
        self.cnt = 0

    def reset(self, total, name=None, **kwargs):
        if self.pbar:
            with self.lock:
                if name:
                    self.pbar.set_description('[%2d] %s' % (self.pos, name))
                self.pbar.reset(total=total)
                self.cnt = 0
        else:
            self.open(name=name, total=total, **kwargs)

    def set_description(self, name):
        with self.lock:
            self.pbar.set_description('[%2d] %s' % (self.pos, name))

    def update(self, inc: int = 1):
        self.cnt += inc
        if self.cnt >= self.update_interval:
            with self.lock:
                self.pbar.update(self.cnt)
            self.cnt = 0

    def close(self):
        with self.lock:
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            if self.pos in self.positions:
                del self.positions[self.pos]

def write_success(filename):
    with open(filename + '._SUCCESS', 'w') as f:
        f.write('%s\n' % datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

def check_success(filename):
    return os.path.exists(filename + '._SUCCESS')