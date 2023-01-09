import os
import re
import torch
import numpy as np
from .dual_position_bert_model import DualPositionBertForPreTrainingPreLN
from .utils import colored, OAG_TOKEN_TYPE_NAMES, stringLenCJK, stringRjustCJK
from transformers import BertTokenizer
import sentencepiece as spm


class OAGMetaInfoBertModel(DualPositionBertForPreTrainingPreLN):
    def __init__(self, bert_config, tokenizer):
        super(OAGMetaInfoBertModel, self).__init__(bert_config)
        self.tokenizer = tokenizer
        self.spm = not isinstance(self.tokenizer, BertTokenizer)
        if self.spm:
            (
                self.tokenizer.cls_token_id,
                self.tokenizer.mask_token_id,
                self.tokenizer.sep_token_id,
            ) = self.tokenizer.PieceToId(["[CLS]", "[MASK]", "[SEP]"])

    def __recursively_build_spm_token_ids(self, text, splitters=[]):
        """
        SentencePiece tokenizer cannot directly decode control symbols such as [MASK] or [SEP]. This function will handle this problem.
        """
        if len(splitters) == 0:
            return self.tokenizer.encode(text)
        splitter = splitters[0]
        splitters = splitters[1:]
        start = 0
        parts = []
        while start >= 0 and start < len(text):
            end = text.find(splitter, start)
            if end >= 0:
                parts += self.__recursively_build_spm_token_ids(text[start:end].strip(), splitters)
                start = end + len(splitter)
                parts.append(self.tokenizer.PieceToId(splitter))
            else:
                end = len(text)
                parts += self.__recursively_build_spm_token_ids(text[start:end].strip(), splitters)
                break
        return parts

    def _convert_text_to_token_ids(self, text):
        if self.spm:
            return self.__recursively_build_spm_token_ids(
                text, splitters=["[PAD]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]"]
            )
        else:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"] if len(text) > 0 else []

    def _convert_ids_to_tokens(self, token_ids):
        if self.spm:
            _ids = []
            for _id in token_ids:
                if not isinstance(_id, int):
                    _id = _id.item()
                _ids.append(_id)
            return self.tokenizer.id_to_piece(_ids)
        else:
            return self.tokenizer.convert_ids_to_tokens(token_ids)

    def _convert_token_ids_to_text(self, token_ids):
        if self.spm:
            _ids = []
            for _id in token_ids:
                if not isinstance(_id, int):
                    _id = _id.item()
                _ids.append(_id)
            return self.tokenizer.decode(_ids)
        else:
            return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token_ids))

    def print_oag_instance(
        self,
        input_ids,
        token_type_ids,
        input_masks,
        masked_lm_labels,
        position_ids,
        position_ids_second,
        predictions=None,
    ):
        COLORS = ["white", "green", "blue", "red", "yellow", "magenta"]
        try:
            termwidth, _ = os.get_terminal_size()
        except Exception:
            termwidth = 200
        K = predictions.shape[1] if predictions is not None else 0
        input_ids = [token_id for i, token_id in enumerate(input_ids) if input_masks[i].sum() > 0]
        position_ids = [position_id for i, position_id in enumerate(position_ids) if input_masks[i].sum() > 0]
        position_ids_second = [
            position_id for i, position_id in enumerate(position_ids_second) if input_masks[i].sum() > 0
        ]
        token_type_ids = [token_type_id for i, token_type_id in enumerate(token_type_ids) if input_masks[i].sum() > 0]
        masks = [0 for i in input_ids]
        prediction_topks = [[0 for i in input_ids] for _ in range(K)]
        mask_indices = []
        for lm_pos, lm_id in enumerate(masked_lm_labels):
            if lm_id < 0:
                continue
            masks[lm_pos] = lm_id
            mask_indices.append(lm_pos)
        for k in range(K):
            for lm_pos, token_id in zip(mask_indices, predictions[:, k]):
                prediction_topks[k][lm_pos] = token_id
        input_tokens = self._convert_ids_to_tokens(input_ids)
        masks_tokens = self._convert_ids_to_tokens(masks)
        prediction_tokens = [self._convert_ids_to_tokens(prediction_topks[k]) for k in range(K)]
        input_tokens_str = [""]
        position_ids_str = [""]
        position_ids_second_str = [""]
        token_type_ids_str = [""]
        masks_str = [""]
        prediction_topk_strs = [[""] for _ in range(K)]
        current_length = 0
        for pos, (input_token, position_id, position_id_second, token_type_id, mask) in enumerate(
            zip(input_tokens, position_ids, position_ids_second, token_type_ids, masks_tokens)
        ):
            token_type = OAG_TOKEN_TYPE_NAMES[token_type_id]
            length = max(
                stringLenCJK(input_token) + 1,
                7,
                stringLenCJK(token_type) + 1,
                stringLenCJK(mask) + 1,
                *[stringLenCJK(prediction_tokens[k][pos]) + 1 for k in range(K)],
            )
            if current_length + length > termwidth:
                current_length = 0
                input_tokens_str.append("")
                position_ids_str.append("")
                position_ids_second_str.append("")
                token_type_ids_str.append("")
                masks_str.append("")
                for k in range(K):
                    prediction_topk_strs[k].append("")
            current_length += length
            input_tokens_str[-1] += colored(stringRjustCJK(input_token, length), COLORS[token_type_id])
            position_ids_str[-1] += stringRjustCJK(str(position_id), length)
            position_ids_second_str[-1] += stringRjustCJK(str(position_id_second), length)
            token_type_ids_str[-1] += stringRjustCJK(token_type, length)
            masks_str[-1] += colored(
                stringRjustCJK(mask, length) if mask != "[PAD]" else stringRjustCJK("", length), COLORS[token_type_id]
            )
            for k in range(K):
                v = prediction_tokens[k][pos] if prediction_tokens[k][pos] != "[PAD]" else ""
                prediction_topk_strs[k][-1] += colored(
                    stringRjustCJK(v, length), "magenta" if v != mask and mask != "[CLS]" else "cyan"
                )

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        size = len(masks_str)
        for i in range(size):
            print()
            print(input_tokens_str[i])
            print(position_ids_str[i])
            print(position_ids_second_str[i])
            print(token_type_ids_str[i])
            if ansi_escape.sub("", masks_str[i]).strip() != "":
                print(masks_str[i])
            for k in range(K):
                if ansi_escape.sub("", prediction_topk_strs[k][i]).strip() != "":
                    print(prediction_topk_strs[k][i])
            print("-" * termwidth)

    def build_inputs(
        self,
        title="",
        abstract="",
        venue="",
        authors=[],
        concepts=[],
        affiliations=[],
        decode_span_type="FOS",
        decode_span_length=0,
        max_seq_length=512,
        mask_propmt_text="",
    ):
        """build inputs from text information for model to use

        Args:
            title (str, optional): [paper title]. Defaults to ''.
            abstract (str, optional): [paper abstract]. Defaults to ''.
            venue (str, optional): [paper venue]. Defaults to ''.
            authors (list, optional): [paper author]. Defaults to [].
            concepts (list, optional): [paper concepts]. Defaults to [].
            affiliations (list, optional): [paper affiliations]. Defaults to [].
            decode_span_type (str, optional): [the span type to decode, choose from 'FOS','VENUE','AFF','AUTHOR']. Defaults to 'FOS'.
            decode_span_length (int, optional): [the length of span to decode]. Defaults to 0.
            max_seq_length (int, optional): [maximum sequence length for the input, the context information will be truncated if the total length exceeds this number]. Defaults to 512.
            mask_propmt_text (str, optional): [the prompt text to add after title and abstract]. Defaults to ''.

        Raises:
            Exception: [provided inputs are invalid]

        Returns:
            [tuple of list]: [input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans]
        """
        input_ids = []
        input_masks = []
        token_type_ids = []
        masked_lm_labels = []
        position_ids = []
        position_ids_second = []
        masked_positions = []
        num_spans = 0

        def add_span(token_type_id, token_ids, is_mask=False):
            nonlocal num_spans
            if len(token_ids) == 0:
                return
            length = len(token_ids)
            input_ids.extend(token_ids if not is_mask else [self.tokenizer.mask_token_id] * length)
            input_masks.extend([1] * length)
            token_type_ids.extend([token_type_id] * length)
            masked_lm_labels.extend([-1] * length if not is_mask else [self.tokenizer.cls_token_id] * length)
            position_ids.extend([num_spans] * length)
            position_ids_second.extend(list(range(length)))
            if is_mask:
                masked_positions.extend([len(input_ids) - length + i for i in range(decode_span_length)])
            num_spans += 1

        span_token_type_id = OAG_TOKEN_TYPE_NAMES.index(decode_span_type)
        if span_token_type_id < 0:
            raise Exception("Unexpected span type: %s" % decode_span_type)

        prompt_token_ids = self._convert_text_to_token_ids(mask_propmt_text)

        add_span(
            0,
            (self._convert_text_to_token_ids(title) + self._convert_text_to_token_ids(abstract) + prompt_token_ids)[
                : max_seq_length - decode_span_length
            ],
        )
        add_span(2, self._convert_text_to_token_ids(venue)[: max_seq_length - len(input_ids) - decode_span_length])
        for author in authors:
            add_span(1, self._convert_text_to_token_ids(author)[: max_seq_length - len(input_ids) - decode_span_length])
        for concept in concepts:
            add_span(
                4, self._convert_text_to_token_ids(concept)[: max_seq_length - len(input_ids) - decode_span_length]
            )
        for affiliation in affiliations:
            add_span(
                3, self._convert_text_to_token_ids(affiliation)[: max_seq_length - len(input_ids) - decode_span_length]
            )

        add_span(span_token_type_id, [0 for _ in range(decode_span_length)], is_mask=True)
        return (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        )

    def encode_paper(
        self,
        title="",
        abstract="",
        venue="",
        authors=[],
        concepts=[],
        affiliations=[],
        decode_span_type="FOS",
        decode_span_length=0,
        max_seq_length=512,
        mask_propmt_text="",
        reduction="first",
    ):
        """encode paper from text information and run forward to get sequence and pool output for each entity

        Args:
            title (str, optional): [paper title]. Defaults to ''.
            abstract (str, optional): [paper abstract]. Defaults to ''.
            venue (str, optional): [paper venue]. Defaults to ''.
            authors (list, optional): [paper author]. Defaults to [].
            concepts (list, optional): [paper concepts]. Defaults to [].
            affiliations (list, optional): [paper affiliations]. Defaults to [].
            decode_span_type (str, optional): [the span type to decode, choose from 'FOS','VENUE','AFF','AUTHOR']. Defaults to 'FOS'.
            decode_span_length (int, optional): [the length of span to decode]. Defaults to 0.
            max_seq_length (int, optional): [maximum sequence length for the input, the context information will be truncated if the total length exceeds this number]. Defaults to 512.
            mask_propmt_text (str, optional): [the prompt text to add after title and abstract]. Defaults to ''.
            reduction (str, optional): [the way to get pooled_output, choose from 'cls','max','mean']. Defaults to 'cls'.

        Raises:
            Exception: [provided inputs are invalid]

        Returns:
            [dictionary of list of dictionary]: {
                'text': text_item,
                'venue': venue_item,
                'authors': [authors_item, authors_item, ...]
                'concepts': [concepts_item, concepts_item, ...]
                'affiliations': [affiliations_item, affiliations_item, ...]
                }
        """
        (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        ) = self.build_inputs(
            title=title,
            abstract=abstract,
            venue=venue,
            authors=authors,
            concepts=concepts,
            affiliations=affiliations,
            decode_span_type=decode_span_type,
            decode_span_length=decode_span_length,
            max_seq_length=max_seq_length,
            mask_propmt_text=mask_propmt_text,
        )

        search = {
            "text": [title + abstract],
            "venue": [venue],
            "authors": authors,
            "concepts": concepts,
            "affiliations": affiliations,
        }

        item = {
            "originalText": "",
            "inputText": "",
            "type": "",
            "tokens": [],
            "token_ids": [],
            "sequence_output": [],
            "pooled_output": [],
        }

        output = {"text": [], "venue": [], "authors": [], "concepts": [], "affiliations": []}

        split_index = {"text": [], "venue": [], "authors": [], "concepts": [], "affiliations": []}

        device = next(self.parameters()).device
        sequence_output, pooled_output = self.bert.forward(
            input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device),
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device),
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device),
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device),
        )

        entities = {0: "text", 2: "venue", 1: "authors", 4: "concepts", 3: "affiliations"}
        for num, name in entities.items():
            if num in token_type_ids:
                start_index = position_ids[token_type_ids.index(num)]
                split_index[name].append(position_ids.index(start_index) - 1)
                for i in range(0, len(search[name])):
                    split_index[name].append(
                        len(position_ids) - 1 - list(reversed(position_ids)).index(start_index + i)
                    )
                    item = item.copy()
                    item["type"] = name.upper()
                    item["originalText"] = search[name][i]
                    item["token_ids"] = input_ids[split_index[name][i] + 1 : split_index[name][i + 1] + 1]
                    item["tokens"] = self._convert_ids_to_tokens(item["token_ids"])
                    item["inputText"] = self._convert_token_ids_to_text(item["token_ids"])
                    item["sequence_output"] = sequence_output[
                        :, split_index[name][i] + 1 : split_index[name][i + 1] + 1, :
                    ].squeeze(0)
                    if reduction == "mean":
                        item["pooled_output"] = item["sequence_output"].mean(dim=0, keepdim=False)
                    elif reduction == "max":
                        item["pooled_output"], _ = item["sequence_output"].max(dim=0)
                    else:
                        item["pooled_output"] = pooled_output
                    output[name].append(item)

        return output

    def calculate_span_prob(
        self,
        title="",
        abstract="",
        venue="",
        authors=[],
        concepts=[],
        affiliations=[],
        decode_span_type="FOS",
        decode_span="",
        force_forward=False,
        max_seq_length=512,
        mask_propmt_text="",
        device=None,
        debug=False,
    ):
        """calculate span probability by greedy algorithm

        Args:
            title (str, optional): [paper title]. Defaults to ''.
            abstract (str, optional): [paper abstract]. Defaults to ''.
            venue (str, optional): [paper venue]. Defaults to ''.
            authors (list, optional): [paper author]. Defaults to [].
            concepts (list, optional): [paper concepts]. Defaults to [].
            affiliations (list, optional): [paper affiliations]. Defaults to [].
            decode_span_type (str, optional): [the span type to decode, choose from 'FOS','VENUE','AFF','AUTHOR']. Defaults to 'FOS'.
            decode_span_length (int, optional): [the length of span to decode]. Defaults to 0.
            force_forward (bool, optional): [if the decoding order is from left to right]. Defaults to False.
            max_seq_length (int, optional): [maximum sequence length for the input, the context information will be truncated if the total length exceeds this number]. Defaults to 512.
            mask_propmt_text (str, optional): [the prompt text to add after title and abstract]. Defaults to ''.
            device ([type], optional): [device for the inputs, default to cpu]. Defaults to None.
            debug (bool, optional): [if debug is true, instances will print to console]. Defaults to False.

        Returns:
            [tuple[float, list of floats]]: [the probability of the whole span, and the individual probability for each token in the decoded span]
        """
        decode_span_token_ids = self._convert_text_to_token_ids(decode_span)
        decode_span_length = len(decode_span_token_ids)
        (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        ) = self.build_inputs(
            title=title,
            abstract=abstract,
            venue=venue,
            authors=authors,
            concepts=concepts,
            affiliations=affiliations,
            decode_span_type=decode_span_type,
            decode_span_length=decode_span_length,
            max_seq_length=max_seq_length,
            mask_propmt_text=mask_propmt_text,
        )

        logprobs = 0.0
        logproblist = []

        def tensorize(x):
            return torch.LongTensor(x).unsqueeze(0).to(device or "cpu")

        for i in range(decode_span_length):
            sequence_output, pooled_output = self.bert.forward(
                input_ids=tensorize(input_ids),
                token_type_ids=tensorize(token_type_ids),
                attention_mask=tensorize(input_masks),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=tensorize(position_ids),
                position_ids_second=tensorize(position_ids_second),
            )
            masked_token_indexes = torch.nonzero((tensorize(masked_lm_labels) + 1).view(-1), as_tuple=False).view(-1)
            prediction_scores, _ = self.cls(sequence_output, pooled_output, masked_token_indexes)
            prediction_scores = torch.nn.functional.log_softmax(prediction_scores, dim=1)  # L x Vocab
            token_log_probs = prediction_scores[torch.arange(len(decode_span_token_ids)), decode_span_token_ids]
            if force_forward:
                logprob, pos = token_log_probs[0], 0
            else:
                logprob, pos = token_log_probs.max(dim=0)
            logprobs += logprob.item()
            logproblist.append(logprob.item())
            real_pos = masked_positions[pos]
            target_token = decode_span_token_ids[pos]
            input_ids[real_pos] = decode_span_token_ids[pos]
            masked_lm_labels[real_pos] = -1
            masked_positions.pop(pos)
            decode_span_token_ids.pop(pos)
            if debug:
                self.print_oag_instance(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    input_masks=input_masks,
                    masked_lm_labels=masked_lm_labels,
                    position_ids=position_ids,
                    position_ids_second=position_ids_second,
                    predictions=torch.topk(prediction_scores, k=5, dim=1).indices.cpu().detach().numpy(),
                )
                input(
                    "logprobs: %.4f, logprob: %.4f, pos: %d, real_pos: %d, token: %s"
                    % (logprobs, logprob, pos.item(), real_pos, self._convert_ids_to_tokens([target_token])[0])
                )

        return np.exp(logprobs), logproblist

    def decode_beamsearch(
        self,
        title="",
        abstract="",
        venue="",
        authors=[],
        concepts=[],
        affiliations=[],
        decode_span_type="",
        decode_span_length=0,
        beam_width=16,
        force_forward=False,
        max_seq_length=512,
        mask_propmt_text="",
        device=None,
        debug=False,
    ):
        """decode span by using beamsearch

        Args:
            title (str, optional): [paper title]. Defaults to ''.
            abstract (str, optional): [paper abstract]. Defaults to ''.
            venue (str, optional): [paper venue]. Defaults to ''.
            authors (list, optional): [paper author]. Defaults to [].
            concepts (list, optional): [paper concepts]. Defaults to [].
            affiliations (list, optional): [paper affiliations]. Defaults to [].
            decode_span_type (str, optional): [the span type to decode, choose from 'FOS','VENUE','AFF','AUTHOR']. Defaults to 'FOS'.
            decode_span_length (int, optional): [the length of span to decode]. Defaults to 0.
            beam_width (int, optional): [beam search width, notice that this function will run one step of beam search in a batch, which should ensure that your gpu (if using) should be able to hold this number of instances]. Defaults to 16.
            force_forward (bool, optional): [if the decoding order is from left to right]. Defaults to False.
            max_seq_length (int, optional): [maximum sequence length for the input, the context information will be truncated if the total length exceeds this number]. Defaults to 512.
            mask_propmt_text (str, optional): [the prompt text to add after title and abstract]. Defaults to ''.
            device ([type], optional): [device for the inputs, default to cpu]. Defaults to None.
            debug (bool, optional): [if debug is true, the beam search progress will be shown]. Defaults to False.

        Returns:
            [list of (string, float)]: [a list of generated spans with their probablities]
        """
        (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        ) = self.build_inputs(
            title=title,
            abstract=abstract,
            venue=venue,
            authors=authors,
            concepts=concepts,
            affiliations=affiliations,
            decode_span_type=decode_span_type,
            decode_span_length=decode_span_length,
            max_seq_length=max_seq_length,
            mask_propmt_text=mask_propmt_text,
        )

        q = [(input_ids, masked_lm_labels, masked_positions, 0)]

        def tensorize(x):
            return torch.LongTensor(x).to(device or "cpu")

        for i in range(decode_span_length):
            sequence_output, pooled_output = self.bert.forward(
                input_ids=tensorize([_input_ids for _input_ids, _, _, _ in q]),
                token_type_ids=tensorize([token_type_ids for _ in q]),
                attention_mask=tensorize([input_masks for _ in q]),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=tensorize([position_ids for _ in q]),
                position_ids_second=tensorize([position_ids_second for _ in q]),
            )
            masked_token_indexes = torch.nonzero(
                (tensorize([_masked_lm_labels for _, _masked_lm_labels, _, _ in q]) + 1).view(-1), as_tuple=False
            ).view(-1)
            prediction_scores, _ = self.cls(sequence_output, pooled_output, masked_token_indexes)
            prediction_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=1
            )  # (len(q) * (range_length - i), VOCAB_SIZE)
            vocab_size = prediction_scores.shape[-1]
            _q = []
            mask_length = decode_span_length - i
            for idx, (_input_ids, _masked_lm_labels, _masked_positions, _last_logprob) in enumerate(q):
                if force_forward:
                    log_probs, indices = torch.topk(prediction_scores[idx * mask_length].view(-1), k=beam_width)
                else:
                    log_probs, indices = torch.topk(
                        prediction_scores[idx * mask_length : (idx + 1) * mask_length].view(-1), k=beam_width
                    )
                for log_prob, index in zip(log_probs.detach().numpy(), indices.detach().numpy()):
                    new_input_ids = _input_ids.copy()
                    new_masked_lm_labels = _masked_lm_labels.copy()
                    new_masked_positions = _masked_positions.copy()
                    fill_id = index % vocab_size
                    rel_fill_pos = index // vocab_size
                    fill_pos = _masked_positions[rel_fill_pos]
                    new_input_ids[fill_pos] = fill_id
                    new_masked_lm_labels[fill_pos] = -1
                    new_masked_positions.pop(rel_fill_pos)
                    _q.append((new_input_ids, new_masked_lm_labels, new_masked_positions, _last_logprob + log_prob))
            _q.sort(key=lambda tup: tup[-1], reverse=True)
            keys = set()
            q.clear()
            for tup in _q:
                key = tuple(tup[0][-decode_span_length:])
                if key not in keys:
                    keys.add(key)
                    q.append(tup)
                    if len(q) >= beam_width:
                        break

            if debug:
                print("beam search, rest length: %d" % (decode_span_length - i))
                for _input_ids, _masked_lm_labels, _masked_positions, _last_logprob in q:
                    print(
                        "  %8.4f, %s"
                        % (_last_logprob, self._convert_token_ids_to_string(_input_ids[-decode_span_length:]),)
                    )

        results = []
        for (_input_ids, _masked_lm_labels, _masked_positions, _logprob) in q:
            generated_entity = self._convert_token_ids_to_text(_input_ids[-decode_span_length:])
            results.append((generated_entity, np.exp(_logprob)))
        return results

    def generate_title(  # noqa C901
        self,
        abstract="",
        authors=[],
        venue="",
        affiliations=[],
        concepts=[],
        num_beams=1,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        min_length=10,
        max_length=30,
        device=None,
        early_stopping=False,
        debug=False,
    ):
        """generate paper titles given other information

        Args:
            abstract (str, optional): [paper abstract]. Defaults to ''.
            venue (str, optional): [paper venue]. Defaults to ''.
            authors (list, optional): [paper author]. Defaults to [].
            affiliations (list, optional): [paper affiliations]. Defaults to [].
            concepts (list, optional): [paper concepts]. Defaults to [].
            num_beams (int, optional): [beam search width, notice that this function will run one step of beam search in a batch, which should ensure that your gpu (if using) should be able to hold this number of instances]. Defaults to 1.
            no_repeat_ngram_size (int, optional): [n-grams phrases cannot repeat in title]. Defaults to 3.
            num_return_sequences (int, optional): [number of sequences to return]. Defaults to 1.
            min_length (int, optional): [the minimum length of generated title]. Defaults to 10.
            min_length (int, optional): [the maximum length of generated title]. Defaults to 30.
            early_stopping (bool, optional): [terminate generation while target number of generated sequences reach <EOS>]. Defaults to false.
            device ([type], optional): [device for the inputs, default to cpu]. Defaults to None.
            debug (bool, optional): [if debug is true, the beam search progress will be shown]. Defaults to False.

        Returns:
            [list of (string, float)]: [a list of generated titles with their probablities]
        """
        if num_return_sequences > num_beams:
            raise Exception(
                "num_return_sequences(%d) cannot be larger than num_beams(%d)" % (num_return_sequences, num_beams)
            )

        selected_ngrams = {}
        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = 1
        token_type_id = 0

        (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        ) = self.build_inputs(
            title="[CLS] [SEP]",
            abstract=abstract,
            venue=venue,
            authors=authors,
            concepts=concepts,
            affiliations=affiliations,
            decode_span_type="TEXT",
            decode_span_length=0,
            max_seq_length=512,
            mask_propmt_text="",
        )

        context_length = len(input_ids)
        num_spans = 0
        decode_pos = 1
        decode_postion_ids_second = 1
        for i in range(1, context_length):
            if token_type_ids[i] == 0:
                position_ids_second[i] = i + 1

        input_ids.insert(decode_pos, mask_token_id)
        token_type_ids.insert(decode_pos, token_type_id)
        position_ids.insert(decode_pos, num_spans)
        position_ids_second.insert(decode_pos, decode_postion_ids_second)
        masked_lm_labels.insert(decode_pos, self.tokenizer.cls_token_id)

        q = [(input_ids, 0)]
        selected_entities = []

        def tensorize(x):
            return torch.LongTensor(x).to(device or "cpu")

        while True:
            batch_input_ids = tensorize([_input_ids for _input_ids, _ in q])
            batch_token_type_ids = tensorize([token_type_ids for _ in q])

            current_total_length = batch_input_ids.shape[1]
            current_entity_length = current_total_length - context_length

            batch_attention_mask = torch.ones((current_total_length, current_total_length))
            batch_attention_mask[
                decode_pos - current_entity_length + 1 : decode_pos + 1,
                decode_pos - current_entity_length + 1 : decode_pos + 1,
            ] = torch.tril(
                batch_attention_mask[
                    decode_pos - current_entity_length + 1 : decode_pos + 1,
                    decode_pos - current_entity_length + 1 : decode_pos + 1,
                ]
            )
            batch_attention_mask = batch_attention_mask.unsqueeze(0).repeat(len(q), 1, 1).to(device or "cpu")

            batch_position_ids = tensorize([position_ids for _ in q])
            batch_position_ids_second = tensorize([position_ids_second for _ in q])
            batch_masked_lm_labels = tensorize([masked_lm_labels for _ in q])
            sequence_output, pooled_output = self.bert.forward(
                input_ids=batch_input_ids,
                token_type_ids=batch_token_type_ids,
                attention_mask=batch_attention_mask,
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=batch_position_ids,
                position_ids_second=batch_position_ids_second,
            )
            masked_token_indexes = torch.nonzero((batch_masked_lm_labels + 1).view(-1)).view(-1)
            prediction_scores, _ = self.cls(sequence_output, pooled_output, masked_token_indexes)
            prediction_scores = torch.nn.functional.log_softmax(prediction_scores, dim=1)
            # surpress existing n-grams
            for idx, (_input_ids, _) in enumerate(q):
                if current_entity_length >= no_repeat_ngram_size:
                    prefix_key = tuple(_input_ids[decode_pos - no_repeat_ngram_size + 1 : decode_pos])
                    for token_id in selected_ngrams.get(prefix_key, set()):
                        prediction_scores[idx, token_id] = -10000
                prefix_key = tuple(_input_ids[decode_pos - current_entity_length : decode_pos])
                if prefix_key in selected_ngrams:
                    for token_id in selected_ngrams.get(prefix_key, set()):
                        prediction_scores[idx, token_id] = -10000
                if current_entity_length <= min_length:
                    prediction_scores[idx, eos_token_id] = -10000
                prediction_scores[idx, _input_ids[decode_pos]] = -10000

            decode_pos += 1
            _q = []
            log_probs, indices = torch.topk(prediction_scores, k=num_beams)
            for idx, (_input_ids, _last_logprob) in enumerate(q):
                for k in range(log_probs.shape[1]):
                    new_input_ids = _input_ids.copy()
                    new_input_ids.insert(decode_pos, indices[idx, k].item())
                    _q.append((new_input_ids, _last_logprob + log_probs[idx, k].item()))

            q = []
            for _input_ids, _last_logprob in _q:
                prefix_key = None
                if current_entity_length >= no_repeat_ngram_size:
                    prefix_key = tuple(_input_ids[decode_pos - no_repeat_ngram_size + 1 : decode_pos])
                    if prefix_key not in selected_ngrams:
                        selected_ngrams[prefix_key] = set()
                    selected_ngrams[prefix_key].add(_input_ids[decode_pos])
                if _input_ids[decode_pos] == eos_token_id:
                    selected_entities.append((_input_ids, _last_logprob))
                else:
                    q.append((_input_ids, _last_logprob))
            q.sort(key=lambda tup: tup[-1], reverse=True)
            selected_entities.sort(key=lambda tup: tup[-1], reverse=True)
            q = q[:num_beams]
            if current_entity_length >= max_length + 2:
                break
            if len(selected_entities) >= num_return_sequences:
                if early_stopping or len(q) == 0 or q[0][-1] <= selected_entities[num_return_sequences - 1][-1]:
                    break

            token_type_ids.insert(decode_pos, token_type_id)
            position_ids.insert(decode_pos, num_spans)
            position_ids_second.insert(decode_pos, decode_postion_ids_second)
            masked_lm_labels[decode_pos - 1] = -1
            masked_lm_labels.insert(decode_pos, self.tokenizer.cls_token_id)

            if debug:
                self.print_oag_instance(
                    input_ids=batch_input_ids[0].cpu().detach().numpy(),
                    token_type_ids=batch_token_type_ids[0].cpu().detach().numpy(),
                    input_masks=batch_attention_mask[0].cpu().detach().numpy(),
                    masked_lm_labels=batch_masked_lm_labels[0].cpu().detach().numpy(),
                    position_ids=batch_position_ids[0].cpu().detach().numpy(),
                    position_ids_second=batch_position_ids_second[0].cpu().detach().numpy(),
                    predictions=torch.topk(prediction_scores, k=5, dim=1).indices.cpu().detach().numpy(),
                )
                input("== Press Enter for next step ==")

        results = []
        for seq, logprob in selected_entities[:num_return_sequences]:
            token_ids = []
            for _id in seq[decode_pos - current_entity_length + 1 : decode_pos]:
                if _id != eos_token_id:
                    token_ids.append(_id)
                else:
                    break
            results.append((self._convert_token_ids_to_text(token_ids), logprob))
        return results
