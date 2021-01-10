import json
import os

import torch
from cogdl.utils import download_url, untar
from transformers import BertTokenizer

from .bert_model import BertConfig, BertForPreTrainingPreLN

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "oagbert-v1": "https://cloud.tsinghua.edu.cn/f/051c9f87d8544698826e/?dl=1",
    "oagbert-test": "https://cloud.tsinghua.edu.cn/f/68a8d42802564d43984e/?dl=1",
}


class OAGBertPretrainingModel(BertForPreTrainingPreLN):
    def __init__(self, bert_config):
        super(OAGBertPretrainingModel, self).__init__(bert_config)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        checkpoint_activations=False,
    ):
        return self.bert.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
        )

    @staticmethod
    def _load(model_name_or_path: str, load_weights: bool = False):
        if not os.path.exists(model_name_or_path):
            if model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
                if not os.path.exists(f"saved/{model_name_or_path}"):
                    archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[model_name_or_path]
                    download_url(archive_file, "saved/", f"{model_name_or_path}.zip")
                    untar("saved/", f"{model_name_or_path}.zip")
                model_name_or_path = f"saved/{model_name_or_path}"
            else:
                raise KeyError("Cannot find the pretrained model {}".format(model_name_or_path))
        bert_config = BertConfig.from_dict(json.load(open(os.path.join(model_name_or_path, "bert_config.json"))))
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        bert_model = OAGBertPretrainingModel(bert_config)

        if load_weights:
            bert_model.load_state_dict(torch.load(os.path.join(model_name_or_path, "pytorch_model.bin")))

        return bert_config, tokenizer, bert_model


def oagbert(model_name_or_path="oagbert-v1", load_weights=True):
    _, tokenizer, bert_model = OAGBertPretrainingModel._load(model_name_or_path, load_weights)

    return tokenizer, bert_model
