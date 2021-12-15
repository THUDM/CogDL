import os
from cogdl.oag import oagbert
import torch
import torch.nn.functional as F
import numpy as np


# load time
tokenizer, model = oagbert("oagbert-v2-sim")
model.eval()

# Paper 1
title = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation..."
authors = ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]
venue = "north american chapter of the association for computational linguistics"
affiliations = ["Google"]
concepts = ["language model", "natural language inference", "question answering"]

# encode first paper
(
    input_ids,
    input_masks,
    token_type_ids,
    masked_lm_labels,
    position_ids,
    position_ids_second,
    masked_positions,
    num_spans,
) = model.build_inputs(
    title=title, abstract=abstract, venue=venue, authors=authors, concepts=concepts, affiliations=affiliations
)
_, paper_embed_1 = model.bert.forward(
    input_ids=torch.LongTensor(input_ids).unsqueeze(0),
    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
    attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
    output_all_encoded_layers=False,
    checkpoint_activations=False,
    position_ids=torch.LongTensor(position_ids).unsqueeze(0),
    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
)

# Positive Paper 2
title = "Attention Is All You Need"
abstract = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely..."
authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"]
venue = "neural information processing systems"
affiliations = ["Google"]
concepts = ["machine translation", "computation and language", "language model"]

(
    input_ids,
    input_masks,
    token_type_ids,
    masked_lm_labels,
    position_ids,
    position_ids_second,
    masked_positions,
    num_spans,
) = model.build_inputs(
    title=title, abstract=abstract, venue=venue, authors=authors, concepts=concepts, affiliations=affiliations
)
# encode second paper
_, paper_embed_2 = model.bert.forward(
    input_ids=torch.LongTensor(input_ids).unsqueeze(0),
    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
    attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
    output_all_encoded_layers=False,
    checkpoint_activations=False,
    position_ids=torch.LongTensor(position_ids).unsqueeze(0),
    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
)

# Negative Paper 3
title = "Traceability and international comparison of ultraviolet irradiance"
abstract = "NIM took part in the CIPM Key Comparison of ″Spectral Irradiance 250 to 2500 nm″. In UV and NIR wavelength, the international comparison results showed that the consistency between Chinese value and the international reference one"
authors = ["Jing Yu", "Bo Huang", "Jia-Lin Yu", "Yan-Dong Lin", "Cai-Hong Dai"]
veune = "Jiliang Xuebao/Acta Metrologica Sinica"
affiliations = ["Department of Electronic Engineering"]
concept = ["Optical Division"]

(
    input_ids,
    input_masks,
    token_type_ids,
    masked_lm_labels,
    position_ids,
    position_ids_second,
    masked_positions,
    num_spans,
) = model.build_inputs(
    title=title, abstract=abstract, venue=venue, authors=authors, concepts=concepts, affiliations=affiliations
)
# encode thrid paper
_, paper_embed_3 = model.bert.forward(
    input_ids=torch.LongTensor(input_ids).unsqueeze(0),
    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
    attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
    output_all_encoded_layers=False,
    checkpoint_activations=False,
    position_ids=torch.LongTensor(position_ids).unsqueeze(0),
    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
)

# calulate text similarity
# normalize
paper_embed_1 = F.normalize(paper_embed_1, p=2, dim=1)
paper_embed_2 = F.normalize(paper_embed_2, p=2, dim=1)
paper_embed_3 = F.normalize(paper_embed_3, p=2, dim=1)

# cosine sim.
sim12 = torch.mm(paper_embed_1, paper_embed_2.transpose(0, 1))
sim13 = torch.mm(paper_embed_1, paper_embed_3.transpose(0, 1))
print(sim12, sim13)
