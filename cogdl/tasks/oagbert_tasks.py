import json
import argparse
import os
from tqdm import tqdm
from langdetect import detect
import random
import time
import torch
import subprocess
import numpy as np
# import multiprocessing
# from multiprocessing import Manager, Pool
from cogdl.oag.bert_model import BertConfig, BertForPreTrainingPreLN
from cogdl.oag.oagbert_metainfo import OAGMetaInfoBertModel
from cogdl.oag.oagbert import OAGBertPretrainingModel
# from cogdl.oag.utils_389 import print_instance, MultiProcessTqdm, write_success, check_success
from transformers import BertTokenizer

from cogdl.utils import download_url, untar
# from cogdl import oagbert
# can import line 71 in oagert.py
from . import BaseTask, register_task

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "oagbert-v1": "https://cloud.tsinghua.edu.cn/f/051c9f87d8544698826e/?dl=1",
    "oagbert-test": "https://cloud.tsinghua.edu.cn/f/68a8d42802564d43984e/?dl=1",
    "oagbert-v2-test": "https://cloud.tsinghua.edu.cn/f/baff5abe84c4483bb690/?dl=1",
    "oagbert-v2": "https://cloud.tsinghua.edu.cn/f/f06448fa3c234317bd16/?dl=1",
}

# class OAGBertPretrainingModel(BertForPreTrainingPreLN):
#     def __init__(self, bert_config):
#         super(OAGBertPretrainingModel, self).__init__(bert_config)

#     def forward(
#         self,
#         input_ids,
#         token_type_ids=None,
#         attention_mask=None,
#         output_all_encoded_layers=False,
#         checkpoint_activations=False,
#     ):
#         return self.bert.forward(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#             attention_mask=attention_mask,
#             output_all_encoded_layers=output_all_encoded_layers,
#             checkpoint_activations=checkpoint_activations,
#         )

#     @staticmethod
#     def _load(model_name_or_path: str, load_weights: bool = False):
#         if not os.path.exists(model_name_or_path):
#             if model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
#                 if not os.path.exists(f"saved/{model_name_or_path}"):
#                     archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[model_name_or_path]
#                     download_url(archive_file, "saved/", f"{model_name_or_path}.zip")
#                     untar("saved/", f"{model_name_or_path}.zip")
#                 model_name_or_path = f"saved/{model_name_or_path}"
#             else:
#                 raise KeyError("Cannot find the pretrained model {}".format(model_name_or_path))

#         try:
#             version = open(os.path.join(model_name_or_path, "version")).readline().strip()
#         except Exception:
#             version = None

#         bert_config = BertConfig.from_dict(json.load(open(os.path.join(model_name_or_path, "bert_config.json"))))
#         tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
#         if version == "2":
#             bert_model = OAGMetaInfoBertModel(bert_config, tokenizer)
#         else:
#             bert_model = OAGBertPretrainingModel(bert_config)

#         model_weight_path = os.path.join(model_name_or_path, "pytorch_model.bin")
#         if load_weights and os.path.exists(model_weight_path):
#             bert_model.load_state_dict(torch.load(model_weight_path))

#         return bert_config, tokenizer, bert_model

# from the code in run_decode_test.py

def get_span_decode_prob(model, tokenizer, model_name, title='', abstract='', venue='', authors=[], concepts=[], affiliations=[], force_forward=False, span_type='', span='', debug=False, max_seq_length=512, device=None, wprop=False, wabs=False):
    token_type_str_lookup = ['TEXT', 'AUTHOR', 'VENUE', 'AFF', 'FOS']
    input_ids = []
    input_masks = []
    token_type_ids = []
    masked_lm_labels = []
    position_ids = []
    position_ids_second = []
    num_spans = 0
    masked_positions = []

    def add_span(token_type_id, token_ids, is_mask=False):
        nonlocal num_spans
        if len(token_ids) == 0:
            return
        length = len(token_ids)
        input_ids.extend(token_ids if not is_mask else [tokenizer.mask_token_id] * length)
        input_masks.extend([1] * length)
        token_type_ids.extend([token_type_id] * length)
        masked_lm_labels.extend([-1] * length if not is_mask else [tokenizer.cls_token_id] * length)
        position_ids.extend([num_spans] * length)
        position_ids_second.extend(list(range(length)))
        if is_mask:
            masked_positions.extend([len(input_ids) - length + i for i in range(span_length)])
        num_spans += 1

    def _encode(text):
        return tokenizer(text, add_special_tokens=False)['input_ids'] if len(text) > 0 else []

    span_token_ids = _encode(span)
    span_length = len(span_token_ids)
    span_token_type_id = token_type_str_lookup.index(span_type)
    if span_token_type_id < 0:
        print('unexpected span type: %s' % span_type)
        return

    prompt_text = ''
    if wprop:
        if span_type == 'FOS':
            prompt_text = 'Field of Study:'
        elif span_type == 'VENUE':
            prompt_text = 'Journal or Venue:'
        elif span_type == 'AFF':
            prompt_text = 'Affiliations:'
        else:
            raise NotImplementedError
    prompt_token_ids = _encode(prompt_text)
    prompt_length = len(prompt_token_ids)

    add_span(0, (_encode(title) + _encode(abstract if wabs else '') + prompt_token_ids)[:max_seq_length - span_length])
    add_span(2, _encode(venue)[:max_seq_length - len(input_ids) - span_length])
    for author in authors:
        add_span(1, _encode(author)[:max_seq_length - len(input_ids) - span_length])
    for concept in concepts:
        add_span(4, _encode(concept)[:max_seq_length - len(input_ids) - span_length])
    for affiliation in affiliations:
        add_span(3, _encode(affiliation)[:max_seq_length - len(input_ids) - span_length])

    add_span(span_token_type_id, span_token_ids, is_mask=True)

    logprobs = 0.
    logproblist = []
    for i in range(span_length):
        if model_name == 'scibert':
            token_type_ids = [0 for _ in token_type_ids]
            position_ids = [idx for idx, _ in enumerate(position_ids)]
            batch = [None] + [torch.LongTensor(t[:max_seq_length]).unsqueeze(0).to(device or 'cpu') for t in [input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second]]
            prediction_logits = model.forward(
                input_ids=batch[1],
                attention_mask=batch[2],
                token_type_ids=batch[3],
                position_ids=batch[5],
            ).prediction_logits
            masked_token_indexes = torch.nonzero((batch[4] + 1).view(-1)).view(-1)
            prediction_scores = prediction_logits.view(-1, prediction_logits.shape[-1])[masked_token_indexes] # L x Vocab
        else:
            batch = [None] + [torch.LongTensor(t[:max_seq_length]).unsqueeze(0).to(device or 'cpu') for t in [input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second]]
            sequence_output, pooled_output = model.bert.forward(
                input_ids=batch[1],
                token_type_ids=batch[3],
                attention_mask=batch[2],
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=batch[5],
                position_ids_second=batch[6])
            masked_token_indexes = torch.nonzero((batch[4] + 1).view(-1)).view(-1)
            prediction_scores, _ = model.cls(sequence_output, pooled_output, masked_token_indexes)
        prediction_scores = torch.nn.functional.log_softmax(prediction_scores, dim=1) # L x Vocab
        token_log_probs = prediction_scores[torch.arange(len(span_token_ids)),span_token_ids]
        if force_forward:
            logprob, pos = token_log_probs[0], 0
        else:
            logprob, pos = token_log_probs.max(dim=0)
        logprobs += logprob.item()
        logproblist.append(logprob.item())
        real_pos = masked_positions[pos]
        target_token = span_token_ids[pos]
        input_ids[real_pos] = span_token_ids[pos]
        masked_lm_labels[real_pos] = -1
        masked_positions.pop(pos)
        span_token_ids.pop(pos)

    return np.exp(logprobs), logproblist


def calculate_samples(model, model_name, tokenizer, input_dir, sample_file, output_file, pbar: tqdm, debug=False, device=None, wprop=False, wabs=False, token_type='FOS', force_forward=False):
    candidates = []
    with open('%s/._SUCCESS' % input_dir) as f:
        for line in f:
            line = line.strip()
            candidates.append(line)

    total = int(subprocess.run(['wc', '-l', sample_file], capture_output=True, text=True).stdout.split(' ')[0]) # 统计行数
    # pbar.reset(total, name=sample_file)
    # pbar.set_description('[%s] loading results from %s' % (sample_file, output_file))
    
    fin = open(sample_file)
    results = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    # pbar.update(1)
                    results.append(json.loads(line.strip()))
                    fin.readline()

    # pbar.set_description('[%s]' % (sample_file))
    fout = open(output_file, 'a')
    for line in fin:
        # pbar.update(1)
        paper = json.loads(line.strip())
        title = paper['title']
        abstract = ''.join(paper['abstracts'])
        obj = {}
        probs = {}
        problists = {}
        for candidate in candidates:
            prob, problist = get_span_decode_prob(model=model, tokenizer=tokenizer, model_name=model_name, title=title, abstract=abstract, span_type=token_type, span=candidate, device=device, debug=debug, wprop=wprop, wabs=wabs, force_forward=force_forward)
            probs[candidate] = prob
            problists[candidate] = problist
        obj['probs'] = list(sorted(probs.items(), key=lambda x: -x[1]))
        obj['pred'] = list(sorted(probs.items(), key=lambda x: -x[1]))[0][0]
        obj['logprobs'] = problists
        results.append(obj)
        fout.write('%s\n' % json.dumps(obj, ensure_ascii=False))

    fin.close()
    fout.close()
    # pbar.close()

def process_file(cuda_idx, input_dir, sample_file, output_file, model_name, wprop=False, wabs=False, token_type='FOS', debug=False, force_forward=False):
    print('process_file called')
    # # if check_success(output_file):
    # #     return
    # # pbar.reset(1, name='loading model')
    # if model_name != 'scibert':
    #     model_weights_path = get_model_weights_path(epoch_name=model_name)
    #     configs = json.load(open('/home/cognitive/pretraining/workspace/oagbert_pretrain/train_metainfo_config.json'))
    #     args, _ = argparse.ArgumentParser().parse_known_args()
    #     args.deepspeed_transformer_kernel = False
    #     args.deepspeed_sparse_attention = False
    #     bert_config = NvidiaBertConfig.from_dict(configs['bert_model_config'])
    #     bert_config.vocab_size = 44000
    #     model = BertForPreTrainingPreLN(bert_config, args)
    #     params = torch.load(model_weights_path, map_location=torch.device('cpu'))['module']
    #     model.load_state_dict(params)
    #     tokenizer = BertTokenizerFast.from_pretrained('/home/cognitive/yinda/oag-redis/data/oagbert_v1')
    # # else:
    # #     model = BertForPreTraining.from_pretrained('/mnt/V2OAG/models/scibert_scivocab_uncased')
    # #     model.cls.predictions.decoder.bias = None
    # #     tokenizer = BertTokenizer.from_pretrained('/mnt/V2OAG/models/scibert_scivocab_uncased')
    # model.eval()
    # device = torch.device('cuda:%d' % cuda_idx if cuda_idx >= 0 else 'cpu')
    # model.to(device)
    # calculate_samples(model=model, model_name=model_name, tokenizer=tokenizer, input_dir=input_dir, sample_file=sample_file, output_file=output_file, pbar=pbar, debug=debug, device=device, wprop=wprop, wabs=wabs, token_type=token_type, force_forward=force_forward)
    # # write_success(output_file)

# def run_mapred(args):

#     # input_dir, output_dir, n_cpu, debug, model_name, wprop, wabs, token_type, force_forward

#     input_dir = args.input_dir
#     os.makedirs(args.output_dir, exist_ok=True)
#     results = []
#     idx = 0
#     for filename in os.listdir(args.input_dir):
#         if not filename.endswith('.jsonl'):
#             continue
#         infile = args.input_dir + '/' + filename
#         outfile = args.output_dir + '/' + filename

#         if not args.debug:
#             process_file(idx % args.n_gpu if not args.debug else -1, args.input_dir, infile, outfile, args.model_name, args.wprop, args.wabs, args.token_type, False, args.force_forward)
#             # results.append((r, filename))
#         else:
#             process_file(idx % args.n_gpu, args.input_dir, infile, outfile, args.model_name, args.wprop, args.wabs, args.token_type, True, args.force_forward)
#         idx += 1
#     # finished = set()
#     # while len(finished) < len(results):
#     #     for r, filename in results:
#     #         if filename not in finished:
#     #             if r.ready():
#     #                 r.get()
#     #                 finished.add(filename)
#     #                 # summary_pbar.update(1)
#     #     time.sleep(1)
#     # pool.close()


@register_task("oagbert_tasks")
class zero_shot_inference(BaseTask):

    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--missing-rate", type=int, default=0, help="missing rate, from 0 to 100")
        # fmt: on
        parser.add_argument('--samples_per_class', type=int, default=1000)
        parser.add_argument('--n_gpu', type=int, default=8)
        parser.add_argument('--wprop', action='store_true', dest='wprop', default=False)
        parser.add_argument('--wabs', action='store_true', dest='wabs', default=False)
        parser.add_argument('--token_type', type=str, default='FOS')
        parser.add_argument('--debug', action='store_true', dest='debug', default=False)
        parser.add_argument('--force_forward', action='store_true', dest='force_forward', default=False)


    def __init__(self, args):
        super(zero_shot_inference, self).__init__(args)

        # required args: dateset, model(SciBert?), task, version?, result-analising?, ...?
        # args include: dataset, model(oagbert (with version?)), task(0shot), 
        '''
        argues to add when run the core function: 
        model, tokenizer, model-name(not necesary), title, abstract, 
        venue, authors=list, concepts=list, affiliations=list, 
        force_forward=False, span_type='', span='', debug=False, max_seq_length=512, device=None, wprop=False, wabs=False
        '''
        self.args = args
        # self.add_args(args)
        print(f'The arguements are: {args}')
        # args: checkpoint=False, cpu=False, dataset='l0fos', debug=False, device_id=[0], eval_step=1, fast_spmm=False, force_forward=False, lr=0.01, max_epoch=500, model='oagbert', n_gpu=8, patience=100, samples_per_class=1000, save_dir='.', seed=1, task='oagbert_tasks', token_type='FOS', trainer=None, use_best_config=False, wabs=False, weight_decay=0.0005, wprop=False

        self.src_dir = 'saved/testset/%s' % args.dataset
        self.input_dir = 'saved/oagbert_v1.5/%s/samples' % args.dataset
        self.output_dir = 'saved/oagbert_v1.5/%s/results/%s-%sprop-%sabs%s%s' % (args.dataset, args.model, 'w' if args.wprop else 'n', 'w' if args.wabs else 'n', '' if not args.force_forward else '_ff', '' if not args.debug else '_debug')
        self.load_model("oagbert-v1", True)

        self.n_gpu = args.n_gpu
        self.debug = args.debug
        self.model_name = args.model
        self.wprop = args.wprop
        self.wabs = args.wprop
        self.token_type = args.token_type
        self.force_forward = args.force_forward
        self.samples_per_class = args.samples_per_class

        print('oagbert')
        self.make_data()

        # if args.model == 'oagbert':
        #     self.model_name = 'oagbert'
        #     self.model, self.tokenizer = oagbert('oagbert-v2')
        
            
    def train(self):
        print('train function called')
        input_dir = self.input_dir
        os.makedirs(self.output_dir, exist_ok=True)
        results = []
        idx = 0
        for filename in os.listdir(self.input_dir):
            if not filename.endswith('.jsonl'):
                continue
            infile = self.input_dir + '/' + filename
            outfile = self.output_dir + '/' + filename

            if not self.debug:
                process_file(idx % self.n_gpu if not self.debug else -1, self.input_dir, infile, outfile, self.model_name, self.wprop, self.wabs, self.token_type, False, self.force_forward)
                # results.append((r, filename))
            else:
                process_file(idx % self.n_gpu, self.input_dir, infile, outfile, self.model_name, self.wprop, self.wabs, self.token_type, True, self.force_forward)
            idx += 1
    
    
    def make_data(self):
        os.makedirs(self.input_dir, exist_ok=True)
        if os.path.exists(self.input_dir + '/._SUCCESS'):
            return
        classes = []
        for filename in os.listdir(self.src_dir):
            if not filename.endswith('.jsonl'):
                continue
            clz = filename.split('.')[0]
            classes.append(clz)
            data = []
            with open('%s/%s' % (self.src_dir, filename)) as f:
                for line in tqdm(f, desc='processing %s' % filename):
                    obj = json.loads(line.strip())
                    try:
                        if detect(obj['title']) != 'en':
                            continue
                        if detect(''.join(obj['abstracts'])) != 'en':
                            continue
                    except Exception as e:
                        print('Error: %s, Obj: %s' % (e, obj))
                        continue
                    data.append(line)
                    if len(data) >= self.samples_per_class:
                        break
            random.shuffle(data)
            if len(data) < self.samples_per_class:
                continue
            with open('%s/%s.jsonl' % (self.input_dir, clz), 'w') as fout:
                for row in data[:self.samples_per_class]:
                    fout.write('%s' % row)
        with open(self.input_dir + '/._SUCCESS', 'w') as f:
            for clz in classes:
                f.write('%s\n' % clz)

    def load_model(self, model_name_or_path: str, load_weights: bool):
        print(model_name_or_path)
        _, self.tokenizer, self.model = OAGBertPretrainingModel._load(model_name_or_path, load_weights)
        print('model loaded')