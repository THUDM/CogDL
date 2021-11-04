import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import logging
from .bert_model import BertPreTrainedModel, BertPreTrainingHeads, BertModel, BertEncoder, BertPooler, BertLayerNorm


logger = logging.getLogger(__name__)


class DualPositionBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(DualPositionBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings_second = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, position_ids, position_ids_second):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings_second = self.position_embeddings(position_ids_second)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + position_embeddings_second + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DualPositionBertModel(BertModel):
    def __init__(self, config):
        super(DualPositionBertModel, self).__init__(config)
        self.embeddings = DualPositionBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        logger.info("Init BERT pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        position_ids=None,
        position_ids_second=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if len(attention_mask.shape) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.shape) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise Exception("invalid attention mask shape! shape: %s" % (attention_mask.shape))

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, position_ids_second)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class DualPositionBertForPreTrainingPreLN(BertPreTrainedModel):
    """BERT model with pre-training heads and dual position
    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    """

    def __init__(self, config):
        super(DualPositionBertForPreTrainingPreLN, self).__init__(config)
        self.bert = DualPositionBertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        position_ids=None,
        position_ids_second=None,
        log=True,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=position_ids,
            position_ids_second=position_ids_second,
        )

        if masked_lm_labels is not None:
            # filter out all masked labels.
            masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1)).view(-1)
            prediction_scores, _ = self.cls(sequence_output, pooled_output, masked_token_indexes)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)
            return masked_lm_loss
        else:
            prediction_scores, _ = self.cls(sequence_output, pooled_output)
            return prediction_scores
