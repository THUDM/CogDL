import json
import os

import torch
from cogdl.utils import download_url, untar
from transformers import BertTokenizer
import sentencepiece as spm

from .bert_model import BertConfig, BertForPreTrainingPreLN
from .oagbert_metainfo import OAGMetaInfoBertModel

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "oagbert-v1": "https://cloud.tsinghua.edu.cn/f/051c9f87d8544698826e/?dl=1",
    "oagbert-test": "https://cloud.tsinghua.edu.cn/f/9277b9229f6246479ec7/?dl=1",
    "oagbert-v2-test": "https://cloud.tsinghua.edu.cn/f/3b8c4525677f4816a138/?dl=1",
    "oagbert-v2": "https://cloud.tsinghua.edu.cn/f/89da6262f3424dd38b05/?dl=1",
    "oagbert-v2-lm": "https://cloud.tsinghua.edu.cn/f/e9e9f435633d4a4ba232/?dl=1",
    "oagbert-v2-sim": "https://cloud.tsinghua.edu.cn/f/e26cb053dbfb45c8af4c/?dl=1",
    "oagbert-v2-zh": "https://cloud.tsinghua.edu.cn/f/cf806c8008b542509201/?dl=1",
    "oagbert-v2-zh-sim": "https://cloud.tsinghua.edu.cn/f/bb6fbc9cda9342698c31/?dl=1",
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

        try:
            version = open(os.path.join(model_name_or_path, "version")).readline().strip()
        except Exception:
            version = None

        bert_config = BertConfig.from_dict(json.load(open(os.path.join(model_name_or_path, "bert_config.json"))))
        if os.path.exists(os.path.join(model_name_or_path, "vocab.txt")):
            tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        elif os.path.exists(os.path.join(model_name_or_path, "vocab.model")):
            tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(model_name_or_path, "vocab.model"))
        else:
            raise FileNotFoundError("Cannot find vocabulary file")
        if version == "2":
            bert_model = OAGMetaInfoBertModel(bert_config, tokenizer)
        else:
            bert_model = OAGBertPretrainingModel(bert_config)

        model_weight_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        if load_weights and os.path.exists(model_weight_path):
            bert_model.load_state_dict(torch.load(model_weight_path))

        return bert_config, tokenizer, bert_model


def oagbert(model_name_or_path="oagbert-v1", load_weights=True):
    """
    load oagbert model, return the underlying torch module and tokenizer. Tokenizer can be either BertTokenizer (en) or SentencePieceTokenizer (zh).
    """
    _, tokenizer, bert_model = OAGBertPretrainingModel._load(model_name_or_path, load_weights)

    return tokenizer, bert_model
