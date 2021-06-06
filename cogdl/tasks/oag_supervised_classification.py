from . import BaseTask, register_task
import argparse
import random
import os
import json
import torch
from collections import namedtuple
from cogdl.oag.oagbert import oagbert
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import sys
import numpy as np
from cogdl.utils import download_url, untar

dataset_url_dict = {
    "l0fos": "https://cloud.tsinghua.edu.cn/f/c2c36282b84043c39ef0/?dl=1",
    "aff30": "https://cloud.tsinghua.edu.cn/f/949c20ff61df469b86d1/?dl=1",
    "arxivvenue": "https://cloud.tsinghua.edu.cn/f/fac19b2aa6a34e9bb176/?dl=1",
}
# python scripts/train.py --task oag_supervised_classification --model oagbert --dataset aff30


class ClassificationModel(torch.nn.Module):
    def __init__(
        self,
        encoder,
        tokenizer,
        num_class,
        device,
        model_name="SciBERT",
        include_fields=["title"],
        max_seq_length=512,
        freeze=False,
    ):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        if freeze:
            for params in self.encoder.parameters():
                params.requires_grad = False
        self.cls = torch.nn.Linear(768, num_class)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.include_fields = include_fields
        self.max_seq_length = max_seq_length
        self.device = device
        self.loss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"] if len(text) > 0 else []

    def build_input(self, sample, labels=None):
        text_input = [self.tokenizer.cls_token_id] + self._encode(
            sample.get("title", "") if "title" in self.include_fields else ""
        )
        if "abstract" in self.include_fields and len(sample.get("abstracts", [])) > 0:
            text_input += [self.tokenizer.sep_token_id] + self._encode(
                "".join(sample.get("abstracts", [])) if "abstract" in self.include_fields else ""
            )
        venue_input = self._encode(sample.get("venue", "") if "venue" in self.include_fields else "")
        aff_input = (
            [self._encode(aff) for aff in sample.get("affiliations", [])] if "aff" in self.include_fields else []
        )
        author_input = (
            [self._encode(author) for author in sample.get("authors", [])] if "author" in self.include_fields else []
        )
        fos_input = [self._encode(fos) for fos in sample.get("fos", [])] if "fos" in self.include_fields else []

        # scibert removed

        input_ids, token_type_ids, position_ids, position_ids_second = [], [], [], []
        entities = (
            [(text_input, 0), (venue_input, 2)]
            + [(_i, 4) for _i in fos_input]
            + [(_i, 3) for _i in aff_input]
            + [(_i, 1) for _i in author_input]
        )
        for idx, (token_ids, token_type_id) in enumerate(entities):
            input_ids += token_ids
            token_type_ids += [token_type_id] * len(token_ids)
            position_ids += [idx] * len(token_ids)
            position_ids_second += list(range(len(token_ids)))
        input_masks = [1] * len(input_ids)
        return input_ids, input_masks, token_type_ids, position_ids, position_ids_second

    def forward(self, samples, labels=None):
        batch = [self.build_input(sample) for sample in samples]
        max_length = min(max(len(tup[0]) for tup in batch), self.max_seq_length)
        padded_inputs = [[] for i in range(4 if self.model_name == "SciBERT" else 5)]
        for tup in batch:
            for idx, seq in enumerate(tup):
                _seq = seq[:max_length]
                _seq += [0] * (max_length - len(_seq))
                padded_inputs[idx].append(_seq)
        input_ids = torch.LongTensor(padded_inputs[0]).to(self.device)
        input_masks = torch.LongTensor(padded_inputs[1]).to(self.device)
        token_type_ids = torch.LongTensor(padded_inputs[2]).to(self.device)
        position_ids = torch.LongTensor(padded_inputs[3]).to(self.device)

        # Only OAGBert available
        position_ids_second = torch.LongTensor(padded_inputs[4]).to(self.device)
        # no degugging
        last_hidden_state, pooled_output = self.encoder.bert.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_masks,
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=position_ids,
            position_ids_second=position_ids_second,
        )
        outputs = self.cls(last_hidden_state.mean(dim=1))  # (B, 768)
        if labels is not None:
            return self.loss(outputs, torch.LongTensor(labels).to(self.device)), outputs.argmax(dim=1)
        else:
            return self.softmax(outputs), outputs.argmax(dim=1)


@register_task("oag_supervised_classification")
class supervised_classification(BaseTask):
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("--include_fields", type=str, nargs="+", default=["title"])
        parser.add_argument("--freeze", action="store_true", default=False)
        parser.add_argument("--cuda", type=int, default=-1)
        parser.add_argument("--testing", action="store_true", default=False)

    def __init__(self, args):
        super().__init__(args)
        self.dataset = args.dataset

        # teporarily fixed constant
        self.testing = args.testing
        self.epochs = 1 if self.testing else 2
        self.batch_size = 16
        self.num_class = 19 if self.dataset == "l0fos" else 30
        self.write_dir = "saved"
        self.cuda = args.cuda
        self.devices = torch.device("cuda:%d" % self.cuda if self.cuda >= 0 else "cpu")

        self.include_fields = args.include_fields
        self.freeze = args.freeze

        self.model = self.load_model()
        self.model.to(self.devices)
        self.train_set, self.dev_set, self.test_set = self.load_data()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.labels = {label: idx for idx, label in enumerate(sorted(set([data["label"] for data in self.train_set])))}

    def load_optimizer(self):
        """
        load the optimizer, self.model required. Learing rate fixed to 2e-5 now.
        """
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=2e-5)

    def load_scheduler(self):
        """
        Load the schedular, self.test_set, self.optimizer, self.epochs, self.batch_size required
        """
        num_train_steps = self.epochs * len(self.train_set) // self.batch_size
        num_warmup_steps = int(num_train_steps * 0.1)
        return get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_train_steps)

    def load_model(self):

        tokenizer, model = oagbert("oagbert-v2", True)
        return ClassificationModel(
            model, tokenizer, self.num_class, self.devices, "OAG-BERT", self.include_fields, 512, self.freeze
        )

    def load_data(self):
        rpath = "data/supervised_classification/" + self.dataset
        zip_name = self.dataset + ".zip"
        if not os.path.isdir(rpath):
            download_url(dataset_url_dict[self.dataset], rpath, name=zip_name)
            untar(rpath, zip_name)

        # dest_dir = '../oagbert/benchmark/raid/yinda/oagbert_v1.5/%s/supervised' % self.dataset
        dest_dir = rpath

        def _load(name):
            data = []
            for line in open("%s/%s.jsonl" % (dest_dir, name)):
                data.append(json.loads(line.strip()))
            return data

        train_data, dev_data, test_data = _load("train"), _load("dev"), _load("test")
        return train_data, dev_data, test_data

    def train(self):
        results = []
        for epoch in range(self.epochs):
            self.run(self.train_set, train=True, shuffle=True, desc="Train %d Epoch" % (epoch + 1))
            score = self.run(self.dev_set, train=False, shuffle=False, desc="Dev %d Epoch" % (epoch + 1))
            torch.save(self.model.state_dict(), self.write_dir + "/Epoch-%d.pt" % (epoch + 1))
            results.append((score, epoch + 1))

        selected_epoch = list(sorted(results, key=lambda t: -t[0]))[0][1]
        self.model.load_state_dict(torch.load(self.write_dir + ("/Epoch-%d.pt" % selected_epoch)))

        return self.test()

    def test(self):
        result = self.run(self.test_set, train=False, shuffle=False, desc="Test")
        for epoch in range(self.epochs):
            os.remove(self.write_dir + "/Epoch-%d.pt" % (epoch + 1))
        return {"Accuracy": result}

    def run(self, dataset, train=False, shuffle=False, desc=""):
        if train:
            self.model.train()
        else:
            self.model.eval()
        if shuffle:
            random.shuffle(dataset)

        size = len(dataset)
        correct, total, total_loss = 0, 0, 0
        pbar = trange(0, size, self.batch_size, ncols=0, desc=desc)
        for i in pbar:
            if self.testing and i % 500 != 0:
                continue
            if train:
                self.optimizer.zero_grad()
            bs = dataset[i : i + self.batch_size]
            y_true = np.array([self.labels[paper["label"]] for paper in bs])
            loss, y_pred = self.model.forward(bs, y_true)
            y_pred = y_pred.cpu().detach().numpy()
            total += len(y_pred)
            correct += (y_pred == y_true).sum()
            total_loss += loss.item()
            if train:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            pbar.set_description("%s Loss: %.4f Acc: %.4f" % (desc, total_loss / total, correct / total))
        pbar.close()
        return correct / total
