import json
import argparse
import os
from tqdm import tqdm
import time
import torch
import numpy as np
from cogdl.oag.oagbert import OAGBertPretrainingModel, oagbert
import multiprocessing
from multiprocessing import Manager
from cogdl.oag.utils import MultiProcessTqdm
from cogdl.datasets import build_dataset
from collections import Counter
from . import BaseTask, register_task

# python scripts/train.py --task oag_zero_shot_infer --model oagbert --dataset l0fos


def get_span_decode_prob(
    model,
    tokenizer,
    title="",
    abstract="",
    venue="",
    authors=[],
    concepts=[],
    affiliations=[],
    span_type="",
    span="",
    debug=False,
    max_seq_length=512,
    device=None,
    wprop=False,
    wabs=False,
    testing=False,
):
    token_type_str_lookup = ["TEXT", "AUTHOR", "VENUE", "AFF", "FOS"]
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
        return tokenizer(text, add_special_tokens=False)["input_ids"] if len(text) > 0 else []

    span_token_ids = _encode(span)
    span_length = len(span_token_ids)
    span_token_type_id = token_type_str_lookup.index(span_type)
    if span_token_type_id < 0:
        print("unexpected span type: %s" % span_type)
        return

    prompt_text = ""
    if wprop:
        if span_type == "FOS":
            prompt_text = "Field of Study:"
        elif span_type == "VENUE":
            prompt_text = "Journal or Venue:"
        elif span_type == "AFF":
            prompt_text = "Affiliations:"
        else:
            raise NotImplementedError
    prompt_token_ids = _encode(prompt_text)

    add_span(0, (_encode(title) + _encode(abstract if wabs else "") + prompt_token_ids)[: max_seq_length - span_length])
    add_span(2, _encode(venue)[: max_seq_length - len(input_ids) - span_length])
    for author in authors:
        add_span(1, _encode(author)[: max_seq_length - len(input_ids) - span_length])
    for concept in concepts:
        add_span(4, _encode(concept)[: max_seq_length - len(input_ids) - span_length])
    for affiliation in affiliations:
        add_span(3, _encode(affiliation)[: max_seq_length - len(input_ids) - span_length])

    add_span(span_token_type_id, span_token_ids, is_mask=True)

    logprobs = 0.0
    logproblist = []
    for i in range(span_length):
        if testing and i % 10 != 0:
            continue
        # scibert deleted
        batch = [None] + [
            torch.LongTensor(t[:max_seq_length]).unsqueeze(0).to(device or "cpu")
            for t in [input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second]
        ]
        sequence_output, pooled_output = model.bert.forward(
            input_ids=batch[1],
            token_type_ids=batch[3],
            attention_mask=batch[2],
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=batch[5],
            position_ids_second=batch[6],
        )
        masked_token_indexes = torch.nonzero((batch[4] + 1).view(-1)).view(-1)
        prediction_scores, _ = model.cls(sequence_output, pooled_output, masked_token_indexes)
        prediction_scores = torch.nn.functional.log_softmax(prediction_scores, dim=1)  # L x Vocab
        token_log_probs = prediction_scores[torch.arange(len(span_token_ids)), span_token_ids]

        # not force forward
        logprob, pos = token_log_probs.max(dim=0)

        logprobs += logprob.item()
        logproblist.append(logprob.item())
        real_pos = masked_positions[pos]
        input_ids[real_pos] = span_token_ids[pos]
        masked_lm_labels[real_pos] = -1
        masked_positions.pop(pos)
        span_token_ids.pop(pos)

    return np.exp(logprobs), logproblist


@register_task("oag_zero_shot_infer")
class zero_shot_inference(BaseTask):
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("--cuda", type=int, nargs="+", default=[-1])
        parser.add_argument("--wprop", action="store_true", dest="wprop", default=False)
        parser.add_argument("--wabs", action="store_true", dest="wabs", default=False)
        parser.add_argument("--token_type", type=str, default="FOS")
        parser.add_argument("--testing", action="store_true", dest="testing", default=False)

    def __init__(self, args):
        super(zero_shot_inference, self).__init__(args)

        self.dataset = build_dataset(args)
        self.sample = self.dataset.get_data()
        self.input_dir = self.dataset.processed_dir
        self.output_dir = "saved/zero_shot_infer/"

        self.tokenizer, self.model = oagbert("oagbert-v2", True)

        self.cudalist = args.cuda
        self.model_name = args.model
        self.wprop = args.wprop  # prompt
        self.wabs = args.wabs  # with abstract
        self.token_type = args.token_type

        self.testing = args.testing

        os.makedirs(self.output_dir, exist_ok=True)
        for filename in os.listdir(self.output_dir):
            os.remove("%s/%s" % (self.output_dir, filename))

    def process_file(self, device, filename, pbar):
        pbar.reset(1, name="preparing...")
        self.model.eval()
        self.model.to(device)

        output_file = self.output_dir + "/" + filename
        candidates = self.dataset.get_candidates()

        pbar.reset(len(self.sample[filename]), name=filename)
        pbar.set_description("[%s]" % (filename))
        fout = open(output_file, "a")
        i = 0
        for paper in self.sample[filename]:
            pbar.update(1)
            i = i + 1
            if self.testing and i % 50 != 0:
                continue
            title = paper["title"]
            abstract = "".join(paper["abstracts"])
            obj, probs, problists = {}, {}, {}
            for candidate in candidates:
                prob, problist = get_span_decode_prob(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    title=title,
                    abstract=abstract,
                    span_type=self.token_type,
                    span=candidate,
                    device=device,
                    debug=False,
                    wprop=self.wprop,
                    wabs=self.wabs,
                    testing=self.testing,
                )
                probs[candidate] = prob
                problists[candidate] = problist
            obj["probs"] = list(sorted(probs.items(), key=lambda x: -x[1]))
            obj["pred"] = list(sorted(probs.items(), key=lambda x: -x[1]))[0][0]
            obj["logprobs"] = problists
            fout.write("%s\n" % json.dumps(obj, ensure_ascii=False))

        fout.close()
        pbar.close()

    def train(self):
        with Manager() as manager:
            lock = manager.Lock()
            positions = manager.dict()

            summary_pbar = MultiProcessTqdm(lock, positions, update_interval=1)
            if -1 not in self.cudalist:
                processnum = 4 * len(self.cudalist)
            else:
                processnum = 12
            pool = multiprocessing.get_context("spawn").Pool(processnum)
            results = []
            idx = 0

            for filename in self.sample.keys():
                if self.testing and idx % 3 != 0:
                    continue
                cuda_num = len(self.cudalist)
                cuda_idx = self.cudalist[idx % cuda_num]
                device = torch.device("cuda:%d" % cuda_idx if cuda_idx >= 0 else "cpu")

                pbar = MultiProcessTqdm(lock, positions, update_interval=1)
                r = pool.apply_async(self.process_file, (device, filename, pbar))
                results.append((r, filename))
                idx += 1

            if self.testing:
                cuda_num = len(self.cudalist)
                cuda_idx = self.cudalist[idx % cuda_num]
                device = torch.device("cuda:%d" % cuda_idx if cuda_idx >= 0 else "cpu")

                pbar = MultiProcessTqdm(lock, positions, update_interval=1)
                for filename in self.sample.keys():
                    self.process_file(device, filename, pbar)
                    break

            summary_pbar.reset(total=len(results), name="Total")
            finished = set()
            while len(finished) < len(results):
                for r, filename in results:
                    if filename not in finished:
                        if r.ready():
                            r.get()
                            finished.add(filename)
                            summary_pbar.update(1)
                time.sleep(1)
            pool.close()
        return self.analysis_result()

    def analysis_result(self):
        concepts = self.dataset.get_candidates()
        concepts.sort()

        result = {}
        T, F = 0, 0
        for filename in os.listdir(self.output_dir):
            if not filename.endswith(".jsonl"):
                continue

            fos = filename.split(".")[0]
            t, f = 0, 0
            cnter = Counter()
            for row in open("%s/%s" % (self.output_dir, filename)):
                try:
                    probs = json.loads(row.strip())["probs"]
                    pred = [
                        k for k, v in sorted([(k, v) for k, v in probs if k in concepts], key=lambda tup: -tup[1])[:2]
                    ]

                except Exception as e:
                    print("Err:%s" % e)
                    print("Row:%s" % row)
                correct = pred[0] == fos
                t += correct
                f += not correct
                cnter[pred[0]] += 1
            T += t
            F += f
            os.remove("%s/%s" % (self.output_dir, filename))
        result["Accuracy"] = T * 100 / (T + F)
        return result
