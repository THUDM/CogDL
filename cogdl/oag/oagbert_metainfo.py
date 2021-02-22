import os
import re
import torch
import numpy as np
from .dual_position_bert_model import DualPositionBertForPreTrainingPreLN
from .utils import colored, OAG_TOKEN_TYPE_NAMES


class OAGMetaInfoBertModel(DualPositionBertForPreTrainingPreLN):

    def __init__(self, bert_config, tokenizer):
        super(OAGMetaInfoBertModel, self).__init__(bert_config)
        self.tokenizer = tokenizer

    def _convert_text_to_token_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)['input_ids'] if len(text) > 0 else []

    def _convert_token_ids_to_text(self, token_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token_ids))

    def print_oag_instance(self,
                           input_ids,
                           token_type_ids,
                           input_masks,
                           masked_lm_labels,
                           position_ids,
                           position_ids_second,
                           predictions=None):
        COLORS = ['white', 'green', 'blue', 'red', 'yellow']
        try:
            termwidth, _ = os.get_terminal_size()
        except Exception:
            termwidth = 200
        K = predictions.shape[1] if predictions is not None else 0
        input_ids = [token_id for i, token_id in enumerate(input_ids) if input_masks[i] > 0]
        position_ids = [position_id for i, position_id in enumerate(position_ids) if input_masks[i] > 0]
        position_ids_second = [position_id for i, position_id in enumerate(position_ids_second) if input_masks[i] > 0]
        token_type_ids = [token_type_id for i, token_type_id in enumerate(token_type_ids) if input_masks[i] > 0]
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
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        masks_tokens = self.tokenizer.convert_ids_to_tokens(masks)
        prediction_tokens = [self.tokenizer.convert_ids_to_tokens(prediction_topks[k]) for k in range(K)]
        input_tokens_str = ['']
        position_ids_str = ['']
        position_ids_second_str = ['']
        token_type_ids_str = ['']
        masks_str = ['']
        prediction_topk_strs = [[''] for _ in range(K)]
        current_length = 0
        for pos, (input_token, position_id, position_id_second, token_type_id, mask) in enumerate(zip(input_tokens, position_ids, position_ids_second, token_type_ids, masks_tokens)):
            token_type = OAG_TOKEN_TYPE_NAMES[token_type_id]
            length = max(len(input_token) + 1, 7, len(token_type) + 1, len(mask)
                         + 1, *[len(prediction_tokens[k][pos]) + 1 for k in range(K)])
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
                prediction_topk_strs[k][-1] += colored(v.rjust(length), 'magenta' if v
                                                       != mask and mask != '[CLS]' else 'cyan')

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

    def build_inputs(self,
                     title='',
                     abstract='',
                     venue='',
                     authors=[],
                     concepts=[],
                     affiliations=[],
                     decode_span_type='FOS',
                     decode_span_length=0,
                     max_seq_length=512,
                     mask_propmt_text=''):
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
            raise Exception('Unexpected span type: %s' % decode_span_type)

        prompt_token_ids = self._convert_text_to_token_ids(mask_propmt_text)

        add_span(0, (self._convert_text_to_token_ids(title)
                     + self._convert_text_to_token_ids(abstract)
                     + prompt_token_ids)[:max_seq_length - decode_span_length])
        add_span(2, self._convert_text_to_token_ids(venue)[:max_seq_length - len(input_ids) - decode_span_length])
        for author in authors:
            add_span(1, self._convert_text_to_token_ids(author)[:max_seq_length - len(input_ids) - decode_span_length])
        for concept in concepts:
            add_span(4, self._convert_text_to_token_ids(concept)[:max_seq_length - len(input_ids) - decode_span_length])
        for affiliation in affiliations:
            add_span(3, self._convert_text_to_token_ids(affiliation)[
                     :max_seq_length - len(input_ids) - decode_span_length])

        add_span(span_token_type_id, [0 for _ in range(decode_span_length)], is_mask=True)
        return input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans

    def calculate_span_prob(self,
                            title='',
                            abstract='',
                            venue='',
                            authors=[],
                            concepts=[],
                            affiliations=[],
                            decode_span_type='FOS',
                            decode_span='',
                            force_forward=False,
                            max_seq_length=512,
                            mask_propmt_text='',
                            device=None,
                            debug=False):
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
        input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = self.build_inputs(
            title=title, abstract=abstract, venue=venue, authors=authors, concepts=concepts,
            affiliations=affiliations, decode_span_type=decode_span_type, decode_span_length=decode_span_length,
            max_seq_length=max_seq_length, mask_propmt_text=mask_propmt_text
        )

        logprobs = 0.
        logproblist = []

        def tensorize(x):
            return torch.LongTensor(x).unsqueeze(0).to(device or 'cpu')
        for i in range(decode_span_length):
            sequence_output, pooled_output = self.bert.forward(
                input_ids=tensorize(input_ids),
                token_type_ids=tensorize(token_type_ids),
                attention_mask=tensorize(input_masks),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=tensorize(position_ids),
                position_ids_second=tensorize(position_ids_second))
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
                self.print_oag_instance(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        input_masks=input_masks,
                                        masked_lm_labels=masked_lm_labels,
                                        position_ids=position_ids,
                                        position_ids_second=position_ids_second,
                                        predictions=torch.topk(prediction_scores, k=5, dim=1).indices.cpu().detach().numpy())
                input('logprobs: %.4f, logprob: %.4f, pos: %d, real_pos: %d, token: %s' %
                      (logprobs, logprob, pos.item(), real_pos, self.tokenizer.convert_ids_to_tokens([target_token])[0]))

        return np.exp(logprobs), logproblist

    def decode_beamsearch(self,
                          title='',
                          abstract='',
                          venue='',
                          authors=[],
                          concepts=[],
                          affiliations=[],
                          decode_span_type='',
                          decode_span_length=0,
                          beam_width=16,
                          force_forward=False,
                          max_seq_length=512,
                          mask_propmt_text='',
                          device=None,
                          debug=False):
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
        input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = self.build_inputs(
            title=title, abstract=abstract, venue=venue, authors=authors, concepts=concepts,
            affiliations=affiliations, decode_span_type=decode_span_type, decode_span_length=decode_span_length,
            max_seq_length=max_seq_length, mask_propmt_text=mask_propmt_text
        )

        q = [(input_ids, masked_lm_labels, masked_positions, 0)]

        def tensorize(x):
            return torch.LongTensor(x).to(device or 'cpu')
        for i in range(decode_span_length):
            sequence_output, pooled_output = self.bert.forward(
                input_ids=tensorize([_input_ids for _input_ids, _, _, _ in q]),
                token_type_ids=tensorize([token_type_ids for _ in q]),
                attention_mask=tensorize([input_masks for _ in q]),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=tensorize([position_ids for _ in q]),
                position_ids_second=tensorize([position_ids_second for _ in q]))
            masked_token_indexes = torch.nonzero(
                (tensorize([_masked_lm_labels for _, _masked_lm_labels, _, _ in q]) + 1).view(-1), as_tuple=False).view(-1)
            prediction_scores, _ = self.cls(sequence_output, pooled_output, masked_token_indexes)
            prediction_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=1)  # (len(q) * (range_length - i), VOCAB_SIZE)
            vocab_size = prediction_scores.shape[-1]
            _q = []
            mask_length = decode_span_length - i
            for idx, (_input_ids, _masked_lm_labels, _masked_positions, _last_logprob) in enumerate(q):
                if force_forward:
                    log_probs, indices = torch.topk(prediction_scores[idx * mask_length].view(-1), k=beam_width)
                else:
                    log_probs, indices = torch.topk(
                        prediction_scores[idx * mask_length:(idx + 1) * mask_length].view(-1), k=beam_width)
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
                print('beam search, rest length: %d' % (decode_span_length - i))
                for _input_ids, _masked_lm_labels, _masked_positions, _last_logprob in q:
                    print('  %8.4f, %s' % (_last_logprob, self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(_input_ids[-decode_span_length:]))))

        results = []
        for (_input_ids, _masked_lm_labels, _masked_positions, _logprob) in q:
            generated_entity = self._convert_token_ids_to_text(_input_ids[-decode_span_length:])
            results.append((generated_entity, np.exp(_logprob)))
        return results
